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

# Teleop key mapping: key → joint and direction
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

# Linear mode boundaries (0–100 scale)
LINEAR_MIN_DEGREE = 0.0
LINEAR_MAX_DEGREE = 100.0