# config.py
import os
import json
import math
from typing import Dict, Tuple, List

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

# Directory to look for saved calibration files
CALIBRATION_FOLDER = "configs"

# Path to default fallback calibration file (used if latest not found)
CALIBRATION_FILENAME = "calib_20250803_100849.json"

# Path to default fallback joint limits file (used if latest not found)
LIMITS_FILENAME = "limits_20250803_100849.json"
# Path to joint limits file
LIMITS_PATH = os.path.join(CALIBRATION_FOLDER, LIMITS_FILENAME)

# Vector order used everywhere (all FK/Jacobian/IK expect this order)
JOINT_ORDER: List[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

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

# --- SERVO CALIBRATION ---

# Helper: Load joint limits (deg)
def _load_joint_limits_from_json(path: str) -> Dict[str, Tuple[float, float]]:
    # Read JSON
    with open(path, "r") as f:
        blob = json.load(f)
    names = blob["motor_names"]
    # Absolute angles (in degrees)
    start_deg = blob["start_deg"]
    end_deg = blob["end_deg"]
    # Create a dictionary
    limits = {}
    for name, a, b in zip(names, start_deg, end_deg):
        lo, hi = (min(a, b), max(a, b))
        limits[name] = (lo, hi)
    return limits
# Load joint limits (deg)
try:
    JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = _load_joint_limits_from_json(LIMITS_PATH)
    if DEBUG:
        print(f"[DEBUG] Loaded joint limits from {LIMITS_PATH}")
except Exception as ex:
    if DEBUG:
        print(f"[WARN] Could not load limits JSON: {ex}\n Falling back to conservative defaults.")
    # Conservative fallback (deg)
    JOINT_LIMITS_DEG = {
        "shoulder_pan": (100.0, 290.0),
        "shoulder_lift": (75.0, 288.0),
        "elbow_flex": (78.0, 272.0),
        "wrist_flex": (71.0, 246.0),
        "wrist_roll": (3.6, 342.5),
        "gripper": (183.0, 280.0),
    }


# --- SO-100 DIMENSIONS ---
# BASE_HEIGHT_M: vertical offset from base frame to shoulder_pitch axis
BASE_HEIGHT_M = 0.070
# L1_UPPER_ARM_M: shoulder_lift link length (shoulder to elbow)
L1_UPPER_ARM_M = 0.160
# L2_FOREARM_M:   elbow_flex link length (elbow to wrist_pitch)
L2_FOREARM_M = 0.160
# L3_WRIST_M:     wrist_pitch link length (wrist_pitch to wrist_roll)
L3_WRIST_M = 0.100
# TOOL_LENGTH_M:  wrist_roll frame to fingertip/end-effector(EE) reference point
TOOL_LENGTH_M = 0.120
# LINK_LENGTHS_M: Tuple
LINK_LENGTHS_M = (BASE_HEIGHT_M, L1_UPPER_ARM_M, L2_FOREARM_M, L3_WRIST_M, TOOL_LENGTH_M)

# --- MODIFIED DENAVIT-HARTENBERG (MDH) TABLE ---
# Degrees to Radians (helper)
def deg(x): return x * math.pi / 180.0
# Constant that maps the servo's "0 reference" to the MDH zero
MDH_THETA_OFFSETS_DEG = {
    "shoulder_pan":   0.0,
    "shoulder_lift":  0.0,
    "elbow_flex":     0.0,
    "wrist_flex":     0.0,
    "wrist_roll":     0.0,
    "gripper":        0.0,
}
# MDH Parameters (a_i, alpha_i, d_i, theta_offset_i) for joints 1 to 6 (in Radians)
MDH_PARAMS = [
    # a_i         alpha_i         d_i                 theta_offset_i (radians)
    (0.000,       +deg(90.0),     BASE_HEIGHT_M,      deg(MDH_THETA_OFFSETS_DEG["shoulder_pan"])),
    (L1_UPPER_ARM_M,  0.000,      0.000,              deg(MDH_THETA_OFFSETS_DEG["shoulder_lift"])),
    (L2_FOREARM_M,    0.000,      0.000,              deg(MDH_THETA_OFFSETS_DEG["elbow_flex"])),
    (L3_WRIST_M,      0.000,      0.000,              deg(MDH_THETA_OFFSETS_DEG["wrist_flex"])),
    (0.000,           +deg(90.0), 0.000,              deg(MDH_THETA_OFFSETS_DEG["wrist_roll"])),
    (0.000,            0.000,     0.000,              deg(MDH_THETA_OFFSETS_DEG["gripper"])),
]

# Tool transform
# Rotate tool about its local X/Y/Z (deg)
TOOL_RPY_DEG = (0.0, 0.0, 0.0)
# Offset along tool axis (assume X-axis)
TOOL_POS_M   = (TOOL_LENGTH_M, 0.0, 0.0)

# --- INVERSE KINEMATICS (IK) ---
# DLS damping
IK_DAMPING_LAMBDA = 1e-2
# Positional tolerance (M)
IK_POS_TOL_M = 1e-3
# Angular tolerance (radians)
IK_ANG_TOL_RAD = deg(0.5)
# Weight for rotational error
IK_ROT_WEIGHT = 1.0
# IK no. of max iterations
IK_MAX_ITERS = 200

# --- FORWARD KINEMATICS ---
# Max degree change (per second)
MAX_DEG_PER_SEC = {
    "shoulder_pan": 90.0,
    "shoulder_lift": 90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 120.0,
    "wrist_roll": 180.0,
    "gripper": 180.0,
}
# Default trajectory time step dt (1 / frequency)
DEFAULT_TRAJ_DT = 0.02
# Default trajectory duration (s)
DEFAULT_TRAJ_DURATION = 6.0
# Default kind of IK trajectory
DEFAULT_TRAJ_KIND = "quintic"  # Either "quintic", "linear", "cubic", or "quintic"

# Helper: Limits in joint order
def limits_for_vector_order() -> List[Tuple[float, float]]:
    # Return per-joint (lo, hi) degrees in JOINT_ORDER
    return [JOINT_LIMITS_DEG[j] for j in JOINT_ORDER]
# Helper: Boolean for if within joint limits
def is_within_limits_deg(q_deg_vec: List[float]) -> bool:
    # Tests if within limits
    for q, (lo, hi) in zip(q_deg_vec, limits_for_vector_order()):
        if not (lo <= q <= hi):
            return False
    return True
