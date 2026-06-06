import serial
import argparse
import sys
import time

parser = argparse.ArgumentParser(description="Biometric Wi-Fi CSI Logger")
parser.add_argument('-p', '--port', type=str, required=True, help="Your COM port (e.g., COM5)")
parser.add_argument('-n', '--name', type=str, required=True, help="Subject name (e.g., abid, stranger_1)")
parser.add_argument('-b', '--baudrate', type=int, default=921600, help="Baudrate (default 921600)")

args = parser.parse_args()

PORT = args.port
SUBJECT_NAME = args.name
BAUDRATE = args.baudrate

CALIB_FILE = f"{SUBJECT_NAME}_calib.csv"
DATA_FILE = f"{SUBJECT_NAME}_data.csv"

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
except Exception as e:
    print(f"Failed to connect to {PORT}. Is the VS Code monitor closed?")
    sys.exit(1)

def record_phase(filename, duration_sec, phase_name):
    print(f"\n--- Starting {phase_name} Phase in 3 seconds... ---")
    time.sleep(3)
    print(f"Recording to {filename} for {duration_sec} seconds. DO NOT STOP...")
    
    end_time = time.time() + duration_sec
    with open(filename, 'w', newline='') as f:
        while time.time() < end_time:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("CSI_DATA"):
                f.write(line + '\n')

print("\n=== STAGE 1: THE TARE WEIGHT (ROOM CALIBRATION) ===")
print("LEAVE THE ROOM NOW! You have 10 seconds.")
for i in range(10, 0, -1):
    print(f"{i}...")
    time.sleep(1)

# Record 10 seconds of the empty room
record_phase(CALIB_FILE, 10, "CALIBRATION")

print("\n\n=== STAGE 2: BIOMETRIC CAPTURE ===")
print("WALK BACK INTO THE ROOM! Walk in random patterns, sit, stand.")
# Record 60 seconds of you moving
record_phase(DATA_FILE, 60, "BIOMETRIC")

print(f"\nSuccess! Saved {CALIB_FILE} and {DATA_FILE}.")
ser.close()