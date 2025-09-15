# config.py

# Debug flag for print messages
DEBUG = True

# Serial port name used to connect to Feetech USB-UART converter
SERVO_PORT = "COM3"

# Baudrate expected by the servos (must match what you configured them with)
SERVO_BAUDRATE = 1000000

# Timeout for serial packet handling (in milliseconds)
TIMEOUT_MS = 1000

# Servo resolution (steps per full rotation)
DEFAULT_RESOLUTION = 4096

# Path to default fallback calibration file (used if latest not found)
CALIBRATION_PATH = "calib_20250803_100849.json"

# Directory to look for automatically saved calibration files
CALIBRATION_FOLDER = "configs"

# Robot arm motors
SERVOS = {
    "shoulder_pan": (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex": (3, "sts3215"),
    "wrist_flex": (4, "sts3215"),
    "wrist_roll": (5, "sts3215"),
    "gripper": (6, "sts3215"),
}

# Fixed joint movement step size (in degrees)
STEP_SIZE = 5.0

# Linear mode boundaries (0–100 scale)
LINEAR_MIN_DEGREE = 0.0
LINEAR_MAX_DEGREE = 100.0

# --- KEYBOARD CONTROL ---
# Teleop key mapping: key to joint and direction
KEYBOARD_BINDINGS = {
    "q": ("shoulder_pan", +1),
    "a": ("shoulder_pan", -1),
    "w": ("shoulder_lift", +1),
    "s": ("shoulder_lift", -1),
    "e": ("elbow_flex", +1),
    "d": ("elbow_flex", -1),
    "r": ("wrist_flex", +1),
    "f": ("wrist_flex", -1),
    "t": ("wrist_roll", +1),
    "g": ("wrist_roll", -1),
    "y": ("gripper", +1),
    "h": ("gripper", -1),
    "z": "quit",
}

# --- PS4 CONTROLLER ---
# DualShock 4 name
PS4_DEVICE_NAME = "Wireless Controller"
# Control loop frequency (Hz)
PS4_LOOP_HZ = 50
# Stick deadzone
PS4_DEADZONE = 0.12
# Max degree change (per second)
PS4_MAX_DEG_PER_SEC = {
    "shoulder_pan": 90.0,
    "shoulder_lift": 90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 120.0,
    "wrist_roll": 180.0,
    "gripper": 180.0,
}
# PS4 Controller key bindings
PS4_BINDINGS = {
    "LX": ("shoulder_pan",   +1.0),
    "LY": ("shoulder_lift",  -1.0),
    "RY": ("elbow_flex",     -1.0),
    "RX": ("wrist_flex",     +1.0),
    "L1": ("wrist_roll",     -0.6),
    "R1": ("wrist_roll",     +0.6),
    "L2": ("gripper",        -1.0),
    "R2": ("gripper",        +1.0),
}