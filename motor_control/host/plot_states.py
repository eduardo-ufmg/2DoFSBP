import argparse, struct, sys, time, csv
from collections import deque
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PACKET_FMT = "<Iffff"  # timestamp(ms), angle(rad), velocity(rad/s), acceleration(rad/s^2), speedCommand
PACKET_SIZE = struct.calcsize(PACKET_FMT)

def wait_for_start(ser, marker="START"):
    line = b""
    while True:
        c = ser.read(1)
        if not c:
            continue
        if c in b"\r\n":
            if line.decode(errors="ignore").strip() == marker:
                # Clear any remaining bytes in the input buffer to ensure clean start
                time.sleep(0.1)  # Give time for any trailing bytes
                ser.reset_input_buffer()
                return
            line = b""
        else:
            line += c

def read_packet(ser):
    buf = ser.read(PACKET_SIZE)
    if len(buf) != PACKET_SIZE:
        return None
    return struct.unpack(PACKET_FMT, buf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--csv", help="Optional output CSV filepath")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print("Waiting for START...")
    wait_for_start(ser)
    print("Streaming...")

    t_ms = deque(maxlen=args.max_points)
    angle = deque(maxlen=args.max_points)
    vel = deque(maxlen=args.max_points)
    acc = deque(maxlen=args.max_points)
    cmd = deque(maxlen=args.max_points)

    fig, axs = None, None
    if not args.no_plot:
        plt.style.use("seaborn-v0_8")
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
        axs[0].set_ylabel("angle [rad]")
        axs[1].set_ylabel("velocity [rad/s]")
        axs[2].set_ylabel("accel [rad/s²]")
        axs[3].set_ylabel("cmd")
        axs[3].set_xlabel("time [s]")
        lines = [ax.plot([], [])[0] for ax in axs]

        def update(_):
            # Drain any available packets quickly
            while ser.in_waiting >= PACKET_SIZE:
                pkt = read_packet(ser)
                if not pkt: break
                ts, a, v, ac, c = pkt
                t_ms.append(ts)
                angle.append(a)
                vel.append(v)
                acc.append(ac)
                cmd.append(c)
            if not t_ms:
                return lines
            t_sec0 = t_ms[0] / 1000.0
            t_sec = [(x/1000.0 - t_sec0) for x in t_ms]
            data_series = [angle, vel, acc, cmd]
            for ln, ds in zip(lines, data_series):
                ln.set_data(t_sec, ds)
            for ax in axs:
                ax.relim()
                ax.autoscale_view()
            return lines

        ani = FuncAnimation(fig, update, interval=100, blit=False)

    try:
        if args.no_plot:
            while True:
                pkt = read_packet(ser)
                if not pkt:
                    continue
                ts, a, v, ac, c = pkt
                print(f"{ts}ms angle={a:.4f} vel={v:.4f} acc={ac:.4f} cmd={c:.3f}")
        else:
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if args.csv and t_ms:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_ms","time_s_rel","angle","velocity","acceleration","speed_command"])
                t0 = t_ms[0]
                for i in range(len(t_ms)):
                    w.writerow([t_ms[i], (t_ms[i]-t0)/1000.0, angle[i], vel[i], acc[i], cmd[i]])
            print(f"Saved {len(t_ms)} samples to {args.csv}")
        ser.close()

if __name__ == "__main__":
    main()
