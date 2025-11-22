import serial, time, struct, csv, matplotlib.pyplot as plt, numpy as np
from scipy.signal import savgol_filter

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
TEST_DATA_LENGTH = 2000
SAMPLE_PERIOD_SEC = 0.01
TIMEOUT_SEC = 2

# Protocol bytes (reuse from experiment.py / Comms.h)
HOST_CHECK_CONNECTION   = b'\x01'
DEVICE_CHECK_CONNECTION = b'\x02'
HOST_START_TEST         = b'\x03'
DEVICE_ACK_START        = b'\x04'
DEVICE_TEST_SUCCESS     = b'\x05'
HOST_REQUEST_DATA       = b'\x06'
DEVICE_DATA_REQUEST_ACK = b'\x07'
DEVICE_DATA_STREAM_START = b'DATA_START'
DEVICE_DATA_STREAM_END   = b'DATA_END'

def _compute_speed(angle_values):
    angle_np = np.asarray(angle_values, dtype=np.float32)
    raw_speed = np.gradient(angle_np, SAMPLE_PERIOD_SEC)
    # window length must be odd and < len(data)
    win = min(len(angle_np)//5*2+1, 101)
    win = max(11, win if win % 2 == 1 else win+1)
    smooth_speed = savgol_filter(angle_np, window_length=win, polyorder=3, deriv=1, delta=SAMPLE_PERIOD_SEC)
    return smooth_speed

def main():
    print("--- Step Response Host ---")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return
    try:
        ser.reset_input_buffer(); ser.reset_output_buffer()
        print("1. Connection check...")
        ser.write(HOST_CHECK_CONNECTION)
        while True:
            r = ser.read(1)
            if r == DEVICE_CHECK_CONNECTION: break
        input("2. Press [Enter] to start step test...")
        print("3. Send start...")
        ser.write(HOST_START_TEST)
        if ser.read(1) != DEVICE_ACK_START:
            print("Start ack failed"); return
        print("   -> Running test (~20s)...")
        ser.timeout = 60
        if ser.read(1) != DEVICE_TEST_SUCCESS:
            print("Test did not succeed"); return
        ser.timeout = TIMEOUT_SEC
        print("4. Request data...")
        ser.write(HOST_REQUEST_DATA)
        if ser.read(1) != DEVICE_DATA_REQUEST_ACK:
            print("Data request ack failed"); return
        if ser.read(len(DEVICE_DATA_STREAM_START)) != DEVICE_DATA_STREAM_START:
            print("Bad data header"); return
        # After header, now only two arrays (input + angle)
        bytes_per_array = TEST_DATA_LENGTH * 4
        raw_input = ser.read(bytes_per_array)
        raw_angle = ser.read(bytes_per_array)
        footer = ser.read(len(DEVICE_DATA_STREAM_END))
        if footer != DEVICE_DATA_STREAM_END: print("Footer mismatch (continuing)")
        if not (len(raw_input)==bytes_per_array and len(raw_angle)==bytes_per_array):
            print("Incomplete data"); return
        fmt = f'<{TEST_DATA_LENGTH}f'
        input_values = struct.unpack(fmt, raw_input)
        angle_values = struct.unpack(fmt, raw_angle)
        speed_values = _compute_speed(angle_values)
        print("5. Save CSV...")
        with open("step_response_data.csv","w",newline='') as f:
            w = csv.writer(f); w.writerow(["Input","Angle","Speed"])
            for i in range(TEST_DATA_LENGTH):
                w.writerow([input_values[i], angle_values[i], float(speed_values[i])])
        print("6. Plot...")
        t = np.arange(TEST_DATA_LENGTH)*SAMPLE_PERIOD_SEC
        step_time = 1.0  # matches controller pre-step duration
        plt.figure(figsize=(10,8))
        plt.subplot(3,1,1)
        plt.plot(t, input_values, label="Input")
        plt.axvline(step_time, color='red', linestyle='--', label='Step')
        plt.ylabel("Input")
        plt.legend(); plt.grid(alpha=0.5)
        plt.subplot(3,1,2)
        plt.plot(t, angle_values, label="Angle")
        plt.axvline(step_time, color='red', linestyle='--')
        plt.ylabel("Angle")
        plt.legend(); plt.grid(alpha=0.5)
        plt.subplot(3,1,3)
        plt.plot(t, speed_values, label="Computed Speed")
        plt.axvline(step_time, color='red', linestyle='--')
        plt.ylabel("Speed")
        plt.xlabel("Time (s)")
        plt.grid(alpha=0.5); plt.legend()
        plt.tight_layout()
        plt.savefig("step_response_plot.png")
        plt.show()
        print("Done.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial closed.")

if __name__ == "__main__":
    main()
