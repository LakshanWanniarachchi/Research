"""
Raw serial capture for hardware evidence.

Opens the port, optionally resets the ESP32 through the USB bridge's RTS line
(the same line esptool uses for "Hard resetting via RTS pin"), and writes every
byte received to a log, line by line, with a host timestamp. Nothing is sent
to the device. Used to record the boot ROM messages (which show whether the
GPIO12 strapping pin broke the boot) and the first lines of the sample stream.

    python capture_serial.py --port COM7 --seconds 8 --reset --out boot.log
"""
import argparse
import datetime
import time

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--reset", action="store_true",
                    help="pulse EN low via RTS before capturing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, args.baud, 0.1
    ser.dtr = False          # keep GPIO0 high: normal boot, not download mode
    ser.rts = False
    ser.open()

    if args.reset:
        ser.rts = True       # EN low
        time.sleep(0.1)
        ser.rts = False      # EN high -> chip boots
    t0 = time.perf_counter()

    lines = []
    buf = b""
    while time.perf_counter() - t0 < args.seconds:
        buf += ser.read(4096)
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            text = raw.rstrip(b"\r").decode("utf-8", errors="replace")
            lines.append(f"[{time.perf_counter() - t0:7.3f}s] {text}")
    ser.close()

    header = [
        f"Serial capture from {args.port} at {args.baud} baud",
        f"Date: {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        f"Duration: {args.seconds:g} s. Reset via RTS: {'yes' if args.reset else 'no'}. "
        "Read-only: nothing was transmitted to the device.",
    ]
    if args.note:
        header.append(args.note)
    header.append("-" * 60)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")
    print(f"{len(lines)} lines -> {args.out}")


if __name__ == "__main__":
    main()
