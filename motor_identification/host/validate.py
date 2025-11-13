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

def run_experiment_and_process_data(inertia_value):
    """
    Runs experiment, downloads data, and computes derivatives 
    to recover Real Torque.
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
        while ser.read(1) != DEVICE_CHECK_CONNECTION:
            pass
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
        print("Waiting for test completion (~41 seconds)...")
        ser.timeout = 60 
        if ser.read(1) != DEVICE_TEST_SUCCESS:
            print("Error: Test failed or timed out.")
            return None
        print("   -> Test completed.")
        ser.timeout = TIMEOUT_SEC 

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
        
        # --- Process Derivatives to find Real Torque ---
        print("Computing Real Torque from Angle data...")
        dt = SAMPLE_PERIOD_SEC
        window_length = 11 
        poly_order = 3

        # 1. Velocity
        df['Velocity'] = savgol_filter(df['Real_Angle'], window_length, poly_order, deriv=1, delta=dt)
        # 2. Acceleration
        df['Acceleration'] = savgol_filter(df['Real_Angle'], window_length, poly_order, deriv=2, delta=dt)
        # 3. Real Torque (Tau = I * alpha)
        df['Real_Torque'] = inertia_value * df['Acceleration']

        filename = 'validation_data.csv'
        df.to_csv(filename, index=False)
        print(f"   -> Data saved to {filename}")
        
        return df

    except Exception as e:
        print(f"Error during experiment: {e}")
        return None
    finally:
        if ser.is_open:
            ser.close()

def load_model_parameters():
    """Loads parameters from JSON. Raises error if not found."""
    if not os.path.exists(MODEL_FILE):
        print(f"\n[ERROR] Model file '{MODEL_FILE}' not found.")
        print("Please run 'estimate.py' first to generate the model parameters.")
        sys.exit(1)

    try:
        with open(MODEL_FILE, 'r') as f:
            data = json.load(f)
            print(f"\n[INFO] Automatically loaded model from '{MODEL_FILE}'")
            return data['slope'], data['intercept'], data['inertia']
    except Exception as e:
        print(f"\n[ERROR] Failed to read '{MODEL_FILE}': {e}")
        sys.exit(1)

def main():
    print("=== System Validation Tool (Torque Domain) ===")
    
    # 1. Load Model First (Need Inertia for data processing)
    slope, intercept, inertia = load_model_parameters()
    print(f"Model: Torque = {slope:.4f} * Input + {intercept:.4f}")
    print(f"Inertia: {inertia:.6e}")

    # 2. Run Experiment & Calculate Real Torque
    df = run_experiment_and_process_data(inertia)
    if df is None:
        return

    # 3. Calculate Predicted Torque
    print("\n--- 2. Model Prediction ---")
    # Model: Torque = K * Input + Offset
    df['Predicted_Torque'] = (slope * df['Input']) + intercept

    # 4. Analysis
    # RMSE on Torque
    rmse = np.sqrt(((df['Real_Torque'] - df['Predicted_Torque']) ** 2).mean())
    
    print("\n--- Validation Results ---")
    print(f"Torque RMSE: {rmse:.4f} N*m")

    # 5. Plotting
    print("Plotting comparison...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Input
    axs[0].plot(df['Time(s)'], df['Input'], color='gray', alpha=0.7)
    axs[0].set_ylabel('Input Signal')
    axs[0].set_title('Validation: Real vs. Model (Torque Domain)')
    axs[0].grid(True)

    # Plot 2: Torque Comparison
    axs[1].plot(df['Time(s)'], df['Real_Torque'], label='Real (derived from Angle)', color='tab:blue', linewidth=1.5, alpha=0.8)
    axs[1].plot(df['Time(s)'], df['Predicted_Torque'], label='Model Prediction', color='tab:orange', linestyle='--', linewidth=2)
    axs[1].set_ylabel('Torque (N*m)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Plot 3: Error Residuals
    error = df['Real_Torque'] - df['Predicted_Torque']
    axs[2].plot(df['Time(s)'], error, color='tab:red')
    axs[2].set_ylabel('Error (N*m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].grid(True)
    axs[2].set_title(f'Torque Prediction Error (RMSE: {rmse:.4f})')

    plt.tight_layout()
    plt.savefig('validation_results.png')
    print("Results saved to validation_results.png. Displaying...")
    plt.show()

if __name__ == "__main__":
    main()