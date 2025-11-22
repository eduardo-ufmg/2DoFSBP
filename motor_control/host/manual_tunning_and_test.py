import serial
import time
import struct
import csv
import matplotlib.pyplot as plt

# --- Configuration ---
SERIAL_PORT = "/dev/ttyUSB0"  # Change as needed
BAUD_RATE = 115200
TEST_DATA_LENGTH = 1024
TIMEOUT_SEC = 2
SAMPLE_PERIOD_SEC = 0.01  # 10 ms

# --- Protocol Definitions (Must match Comms.h) ---
HOST_CHECK_CONNECTION = b"\x01"
DEVICE_CHECK_CONNECTION = b"\x02"
HOST_START_TEST = b"\x03"
DEVICE_ACK_START = b"\x04"
DEVICE_TEST_SUCCESS = b"\x05"
HOST_REQUEST_DATA = b"\x06"
DEVICE_DATA_REQUEST_ACK = b"\x07"

DEVICE_DATA_STREAM_START = b"DATA_START"
DEVICE_DATA_STREAM_END = b"DATA_END"


def main():
    print("--- Motor PID Test ---")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        time.sleep(2)  # Wait for Arduino to reset after serial connection
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        return

    try:
        # 0. Flush any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # 1. Check connection with controller
        print("1. Checking connection with controller...")
        ser.write(HOST_CHECK_CONNECTION)

        print(f"Waiting for device response...")
        response = b""
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
        # Note: The C++ loop runs for ~10 seconds (1024 * 10ms).
        # We temporarily increase timeout to avoid giving up too early.
        ser.timeout = 120
        print("5. Waiting for test completion (approx. 10-12 seconds)...")

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
        # Expect: "DATA_START" -> [Reference Floats] -> [Effort Floats] -> [Torque Floats] -> "DATA_END"

        # Check header
        header = ser.read(len(DEVICE_DATA_STREAM_START))
        if header != DEVICE_DATA_STREAM_START:
            print(f"Error: Invalid data header. Received: {header}")
            return

        # Calculate bytes to read: 4096 floats * 4 bytes/float
        bytes_per_array = TEST_DATA_LENGTH * 4

        print(f"   -> Reading {TEST_DATA_LENGTH} Reference samples...")
        raw_reference_data = ser.read(bytes_per_array)
        if len(raw_reference_data) != bytes_per_array:
            print(
                f"Error: Incomplete reference data. Got {len(raw_reference_data)} bytes."
            )
            return

        print(f"   -> Reading {TEST_DATA_LENGTH} Effort samples...")
        raw_effort_data = ser.read(bytes_per_array)
        if len(raw_effort_data) != bytes_per_array:
            print(f"Error: Incomplete effort data. Got {len(raw_effort_data)} bytes.")
            return

        print(f"   -> Reading {TEST_DATA_LENGTH} Torque samples...")
        raw_torque_data = ser.read(bytes_per_array)
        if len(raw_torque_data) != bytes_per_array:
            print(f"Error: Incomplete torque data. Got {len(raw_torque_data)} bytes.")
            return

        # Check footer
        footer = ser.read(len(DEVICE_DATA_STREAM_END))
        if footer != DEVICE_DATA_STREAM_END:
            print(f"Warning: Invalid data footer. Received: {footer}")
            # We don't abort here, as we might have valid data anyway.

        # Unpack binary data to float lists
        # '<' = little-endian (standard for ESP32), 'f' = float
        fmt = f"<{TEST_DATA_LENGTH}f"
        reference_values = struct.unpack(fmt, raw_reference_data)
        effort_values = struct.unpack(fmt, raw_effort_data)
        torque_values = struct.unpack(fmt, raw_torque_data)

        # 9. Save data to file
        filename = "test_data.csv"
        print(f"9. Saving data to {filename}...")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Reference", "Effort", "Torque"])
            for i in range(TEST_DATA_LENGTH):
                writer.writerow(
                    [reference_values[i], effort_values[i], torque_values[i]]
                )

        # 10. Report experiment success
        print("10. Experiment finished successfully.")

        # 11. Plot data
        print("11. Plotting results...")
        plot_filename = "test_results.png"

        plt.figure(figsize=(10, 8))

        time_values = [i * SAMPLE_PERIOD_SEC for i in range(TEST_DATA_LENGTH)]

        # Top subplot: Reference and Measured Torque (superposed)
        plt.subplot(2, 1, 1)
        plt.plot(
            time_values,
            reference_values,
            color="blue",
            label="Reference (Torque Setpoint)",
            linewidth=2,
        )
        plt.plot(
            time_values,
            torque_values,
            color="green",
            label="Torque (Measured)",
            linewidth=1.5,
            alpha=0.8,
        )
        plt.title("Motor Experiment Results")
        plt.ylabel("Torque Value")
        plt.grid(True, alpha=0.5)
        plt.legend(loc="upper right")

        # Bottom subplot: Effort
        plt.subplot(2, 1, 2)
        plt.plot(
            time_values, effort_values, color="orange", label="Effort (Motor Command)"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Effort Value")
        plt.grid(True, alpha=0.5)
        plt.legend(loc="upper right")

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
