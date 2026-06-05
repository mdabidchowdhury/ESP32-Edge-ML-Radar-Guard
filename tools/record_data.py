import serial
import argparse
import sys

# Set up argument parsing to make it command-line friendly
parser = argparse.ArgumentParser(description="Lightweight Wi-Fi CSI Logger")
parser.add_argument('-p', '--port', type=str, required=True, help="Your COM port (e.g., COM5)")
parser.add_argument('-s', '--save', type=str, required=True, help="Filename to save (e.g., empty_room.csv)")
parser.add_argument('-b', '--baudrate', type=int, default=115200, help="Baudrate (default 115200)")

args = parser.parse_args()

PORT = args.port
FILENAME = args.save
BAUDRATE = args.baudrate

print(f"Connecting to {PORT}...")
try:
    # Open the serial connection
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
except Exception as e:
    print(f"Failed to connect to {PORT}. Make sure the VS Code IDF Monitor is CLOSED!")
    sys.exit(1)

print(f"Connected! Recording data to {FILENAME}...")
print("Press Ctrl+C when you are finished.")

try:
    with open(FILENAME, 'w', newline='') as f:
        while True:
            # Read the raw serial line and decode it
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # Filter out random boot logs and only save the actual wave data
            if line.startswith("CSI_DATA"):
                f.write(line + '\n')
                
                # Print a dot to the console so you know it's working
                print(".", end="", flush=True)
                
except KeyboardInterrupt:
    print(f"\nSuccessfully saved {FILENAME}!")
    ser.close()