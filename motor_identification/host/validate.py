import serial
import time
import struct
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import savgol_filter

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200
TEST_DATA_LENGTH = 4096
TIMEOUT_SEC = 2
SAMPLE_PERIOD_SEC = 0.01
MODEL_FILE = 'model_parameters.json'

# --- Protocol Definitions ---
HOST_CHECK_CONNECTION   = b'\x01'
DEVICE_CHECK_CONNECTION = b'\x02'
HOST_START_TEST         = b'\x03'
DEVICE_ACK_START        = b'\x04'
DEVICE_TEST_SUCCESS     = b'\x05'
HOST_REQUEST_DATA       = b'\x06'
DEVICE_DATA_REQUEST_ACK = b'\x07'
DEVICE_DATA_STREAM_START = b'DATA_START'
DEVICE_DATA_STREAM_END   = b'DATA_END'

def run_experiment_and_process_data():
    """
    Runs a new experiment, downloads data, and computes derivatives.
    """
    print("\n--- 1. Hardware Interface ---")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        time.sleep(2) 
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        return None

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # 1. Check connection
        print("Connecting to controller...")
        ser.write(HOST_CHECK_CONNECTION)
        
        print(f"Waiting for device response...")
        response = b''
        while response != DEVICE_CHECK_CONNECTION:
            response = ser.read(1)
        print("   -> Connection confirmed.")

        # 2. User Start
        input("Press [Enter] to start the validation experiment...")

        # 3. Start Test
        print("Sending start command...")
        ser.write(HOST_START_TEST)

        if ser.read(1) != DEVICE_ACK_START:
            print("Error: Device did not acknowledge start.")
            return None
        print("   -> Test running...")

        # 4. Wait for completion
        print(f"Waiting for test completion (approx. {TEST_DATA_LENGTH * SAMPLE_PERIOD_SEC:.0f} seconds)...")
        ser.timeout = 60 # Long timeout for test duration
        
        if ser.read(1) != DEVICE_TEST_SUCCESS:
            print("Error: Test failed or timed out.")
            return None
        print("   -> Test completed.")
        ser.timeout = TIMEOUT_SEC # Reset to normal timeout

        # 5. Request Data
        print("Requesting data...")
        ser.write(HOST_REQUEST_DATA)

        if ser.read(1) != DEVICE_DATA_REQUEST_ACK:
            print("Error: Device did not ack data request.")
            return None

        # 6. Read Data
        print("Downloading data stream...")
        if ser.read(len(DEVICE_DATA_STREAM_START)) != DEVICE_DATA_STREAM_START:
            print("Error: Invalid header.")
            return None

        bytes_to_read = TEST_DATA_LENGTH * 4
        raw_input = ser.read(bytes_to_read)
        raw_angle = ser.read(bytes_to_read)

        if len(raw_input) != bytes_to_read or len(raw_angle) != bytes_to_read:
            print("Error: Incomplete data stream.")
            return None

        if ser.read(len(DEVICE_DATA_STREAM_END)) != DEVICE_DATA_STREAM_END:
            print("Warning: Invalid footer.")

        fmt = f'<{TEST_DATA_LENGTH}f'
        input_values = struct.unpack(fmt, raw_input)
        angle_values = struct.unpack(fmt, raw_angle)
        
        # Create DataFrame
        time_axis = [i * SAMPLE_PERIOD_SEC for i in range(TEST_DATA_LENGTH)]
        df = pd.DataFrame({
            'Time(s)': time_axis,
            'Input': input_values,
            'Real_Angle': angle_values
        })
        
        # --- Process Derivatives from new data ---
        print("Computing derivatives from Angle data...")
        dt = SAMPLE_PERIOD_SEC
        
        # Use same filter settings as estimate.py for consistency
        window_length = 51 
        poly_order = 3

        df['Real_Velocity'] = savgol_filter(df['Real_Angle'], window_length, poly_order, deriv=1, delta=dt)
        df['Real_Acceleration'] = savgol_filter(df['Real_Angle'], window_length, poly_order, deriv=2, delta=dt)

        filename = 'validation_data.csv'
        df.to_csv(filename, index=False)
        print(f"   -> Validation data saved to {filename}")
        
        return df

    except Exception as e:
        print(f"Error during experiment: {e}")
        return None
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

def load_model_parameters():
    """Loads dynamic model parameters from JSON."""
    if not os.path.exists(MODEL_FILE):
        print(f"\n[ERROR] Model file '{MODEL_FILE}' not found.")
        print("Please run 'estimate.py' first to generate the model parameters.")
        sys.exit(1)

    try:
        with open(MODEL_FILE, 'r') as f:
            data = json.load(f)
        
        params = data['parameters']
        K_m = params['K_m_gain']
        T_m = params['T_m_time_constant']
        Intercept = params['intercept']
        
        print(f"\n[INFO] Automatically loaded model from '{MODEL_FILE}'")
        return K_m, T_m, Intercept
        
    except KeyError:
        print(f"\n[ERROR] Model file '{MODEL_FILE}' is missing required parameters.")
        print("Expected 'parameters' -> 'K_m_gain', 'T_m_time_constant', 'intercept'")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to read '{MODEL_FILE}': {e}")
        sys.exit(1)

def main():
    print("=== System Validation Tool (Dynamic Velocity Model) ===")
    
    # 1. Load Model
    K_m, T_m, Intercept = load_model_parameters()
    print(f"Model: Velocity = {K_m:.4f} * Input - {T_m:.4f} * Acceleration + {Intercept:.4f}")

    # 2. Run Experiment & Calculate Real Derivatives
    df = run_experiment_and_process_data()
    if df is None:
        print("Experiment failed. Exiting.")
        return

    # 3. Calculate Predicted Velocity using the model
    print("\n--- 2. Model Prediction ---")
    # Model: Velocity = K_m * Input - T_m * Acceleration + Intercept
    df['Predicted_Velocity'] = (K_m * df['Input']) - (T_m * df['Real_Acceleration']) + Intercept
    print("Predicted velocity calculated from Input and Real Acceleration.")

    # 4. Analysis
    # RMSE on Velocity
    rmse = np.sqrt(((df['Real_Velocity'] - df['Predicted_Velocity']) ** 2).mean())
    
    print("\n--- Validation Results ---")
    print(f"Velocity RMSE: {rmse:.6f} rad/s")

    # 5. Plotting
    print("Plotting comparison...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1: Input
    axs[0].plot(df['Time(s)'], df['Input'], color='gray', alpha=0.7)
    axs[0].set_ylabel('Input Signal')
    axs[0].set_title('Validation: Real vs. Model (Velocity Domain)')
    axs[0].grid(True)

    # Plot 2: Velocity Comparison
    axs[1].plot(df['Time(s)'], df['Real_Velocity'], label='Real (derived from Angle)', color='tab:blue', linewidth=1.5, alpha=0.8)
    axs[1].plot(df['Time(s)'], df['Predicted_Velocity'], label='Model Prediction', color='tab:orange', linestyle='--', linewidth=2)
    axs[1].set_ylabel('Velocity (rad/s)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Plot 3: Error Residuals
    error = df['Real_Velocity'] - df['Predicted_Velocity']
    axs[2].plot(df['Time(s)'], error, color='tab:red')
    axs[2].set_ylabel('Velocity Error (rad/s)')
    axs[2].set_xlabel('Time (s)')
    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].grid(True)
    axs[2].set_title(f'Velocity Prediction Error (RMSE: {rmse:.6f})')

    plt.tight_layout()
    plt.savefig('validation_results.png')
    print("Results saved to validation_results.png. Displaying...")
    plt.show()

if __name__ == "__main__":
    main()