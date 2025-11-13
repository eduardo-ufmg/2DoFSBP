import serial
import time
import struct
import csv
import matplotlib.pyplot as plt

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0' # Change as needed
BAUD_RATE = 115200
TEST_DATA_LENGTH = 4096
TIMEOUT_SEC = 2
SAMPLE_PERIOD_SEC = 0.01 # 10 ms

# --- Protocol Definitions (Must match Comms.h) ---
HOST_CHECK_CONNECTION   = b'\x01'
DEVICE_CHECK_CONNECTION = b'\x02'
HOST_START_TEST         = b'\x03'
DEVICE_ACK_START        = b'\x04'
DEVICE_TEST_SUCCESS     = b'\x05'
HOST_REQUEST_DATA       = b'\x06'
DEVICE_DATA_REQUEST_ACK = b'\x07'

DEVICE_DATA_STREAM_START = b'DATA_START'
DEVICE_DATA_STREAM_END   = b'DATA_END'

def main():
    print("--- Motor Control Experiment Host ---")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        time.sleep(2) # Wait for Arduino to reset after serial connection
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        return

    try:
        # 0. Flush any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # 1. Check connection with controller
        print("1. Checking connection with controller...")
        ser.reset_input_buffer()
        ser.write(HOST_CHECK_CONNECTION)
        
        print(f"Waiting for device response...")
        response = b''
        while response != DEVICE_CHECK_CONNECTION:
            response = ser.read(1)
        print("   -> Connection confirmed.")

        # 2. Wait for user to start experiment
        input("2. Press [Enter] to start the experiment...")

        # 3. Send starting message to controller
        print("3. Sending start command...")
        ser.write(HOST_START_TEST)

        # 4. Wait for controller to acknowledge
        response = ser.read(1)
        if response != DEVICE_ACK_START:
            print(f"Error: Device did not acknowledge start. Received: {response}")
            return
        print("   -> Start acknowledged. Test running...")

        # 5. Wait for controller to send success message
        # Note: The C++ loop runs for ~41 seconds (4096 * 10ms). 
        # We temporarily increase timeout to avoid giving up too early.
        ser.timeout = 120 
        print("5. Waiting for test completion (approx. 40-45 seconds)...")
        
        response = ser.read(1)
        if response != DEVICE_TEST_SUCCESS:
            print(f"Error: Test failed or timed out. Received: {response}")
            return
        print("   -> Test completed successfully.")
        
        # Reset timeout to normal for data transfer
        ser.timeout = TIMEOUT_SEC 

        # 6. Request data from controller
        print("6. Requesting data...")
        ser.write(HOST_REQUEST_DATA)

        # 7. Wait for data request ack
        response = ser.read(1)
        if response != DEVICE_DATA_REQUEST_ACK:
            print(f"Error: Device did not ack data request. Received: {response}")
            return
        print("   -> Data request acknowledged. Receiving stream...")

        # 8. Read data from controller
        # Expect: "DATA_START" -> [Input Floats] -> [Angle Floats] -> "DATA_END"
        
        # Check header
        header = ser.read(len(DEVICE_DATA_STREAM_START))
        if header != DEVICE_DATA_STREAM_START:
            print(f"Error: Invalid data header. Received: {header}")
            return

        # Calculate bytes to read: 4096 floats * 4 bytes/float
        bytes_per_array = TEST_DATA_LENGTH * 4
        
        print(f"   -> Reading {TEST_DATA_LENGTH} Input samples...")
        raw_input_data = ser.read(bytes_per_array)
        if len(raw_input_data) != bytes_per_array:
            print(f"Error: Incomplete input data. Got {len(raw_input_data)} bytes.")
            return

        print(f"   -> Reading {TEST_DATA_LENGTH} Angle samples...")
        raw_angle_data = ser.read(bytes_per_array)
        if len(raw_angle_data) != bytes_per_array:
            print(f"Error: Incomplete angle data. Got {len(raw_angle_data)} bytes.")
            return

        # Check footer
        footer = ser.read(len(DEVICE_DATA_STREAM_END))
        if footer != DEVICE_DATA_STREAM_END:
            print(f"Warning: Invalid data footer. Received: {footer}")
            # We don't abort here, as we might have valid data anyway.

        # Unpack binary data to float lists
        # '<' = little-endian (standard for ESP32), 'f' = float
        fmt = f'<{TEST_DATA_LENGTH}f'
        input_values = struct.unpack(fmt, raw_input_data)
        angle_values = struct.unpack(fmt, raw_angle_data)

        # 9. Save data to file
        filename = "experiment_data.csv"
        print(f"9. Saving data to {filename}...")

        time_axis = [i * SAMPLE_PERIOD_SEC for i in range(TEST_DATA_LENGTH)]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time(s)", "Input", "Angle"])
            for i in range(TEST_DATA_LENGTH):
                writer.writerow([time_axis[i], input_values[i], angle_values[i]])

        # 10. Report experiment success
        print("10. Experiment finished successfully.")

        # 11. Plot data
        print("11. Plotting results...")
        plot_filename = "experiment_results.png"
        
        plt.figure(figsize=(10, 8))

        # Top subplot: Input
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, input_values, color='blue', label='Input (Speed Setpoint)')
        plt.title('Motor Experiment Results')
        plt.ylabel('Input Value')
        plt.grid(True, alpha=0.5)
        plt.legend(loc='upper right')

        # Bottom subplot: Angle
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, angle_values, color='orange', label='Measured Angle')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle')
        plt.grid(True, alpha=0.5)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"    -> Plot saved to {plot_filename}")
        
        print("    -> Displaying plot (close window to exit)...")
        plt.show()

        print("Done.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    main()