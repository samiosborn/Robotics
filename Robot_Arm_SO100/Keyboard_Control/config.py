# config.py
import os
import json
import math
from typing import Dict, Tuple, List

# Degrees of Freedom
DOF = 6

# Debug flag for print messages
DEBUG = True

# --- SERVO SETUP ---
# Serial port name used to connect to Feetech USB-UART converter
SERVO_PORT = "COM3"
# Baudrate expected by the servos (must match what you configured them with)
SERVO_BAUDRATE = 1000000
# Timeout for serial packet handling (in milliseconds)
TIMEOUT_MS = 1000
# Servo resolution (steps per full rotation)
DEFAULT_RESOLUTION = 4096

# --- SERVO CALIB & LIMITS ---
# Directory to look for saved calibration files
CALIBRATION_FOLDER = "configs"
# Filename of default calibration file
CALIBRATION_FILENAME = "calib_20250803_100849.json"
# Path to joint calibration file 
CALIBRATION_PATH = os.path.join(CALIBRATION_FOLDER, CALIBRATION_FILENAME)
# Filename of default joint limits file
LIMITS_FILENAME = "limits_20250803_100849.json"
# Path to joint limits file
LIMITS_PATH = os.path.join(CALIBRATION_FOLDER, LIMITS_FILENAME)
# Filename of waypoints (default)
WAYPOINTS_FILENAME = "waypoints.json"
# Path to waypoints file
WAYPOINTS_PATH = os.path.join(CALIBRATION_FOLDER, WAYPOINTS_FILENAME)

# --- SO-100 DIMENSIONS ---
SHOULDER_OFFSET_X_M = 0.050  # +x from base pan axis to shoulder-lift axis
SHOULDER_OFFSET_Z_M = 0.060  # +z from base pan axis to shoulder-lift axis
# L1_UPPER_ARM_M: shoulder_lift link length (shoulder to elbow)
L1_UPPER_ARM_M = 0.160
# L2_FOREARM_M:   elbow_flex link length (elbow to wrist_pitch)
L2_FOREARM_M = 0.160
# L3_WRIST_M:     wrist_pitch link length (wrist_pitch to wrist_roll)
L3_WRIST_M = 0.100
# TOOL_LENGTH_M:  wrist_roll frame to fingertip/end-effector(EE) reference point
TOOL_LENGTH_M = 0.120
# LINK_LENGTHS_M: Tuple
LINK_LENGTHS_M = (SHOULDER_OFFSET_X_M, SHOULDER_OFFSET_Z_M, L1_UPPER_ARM_M, L2_FOREARM_M, L3_WRIST_M, TOOL_LENGTH_M)

# --- SERVO JOINTS ---
# Joint order as vector
JOINT_ORDER: List[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# Robot arm motor name
SERVOS = {
    "shoulder_pan": (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex": (3, "sts3215"),
    "wrist_flex": (4, "sts3215"),
    "wrist_roll": (5, "sts3215"),
    "gripper": (6, "sts3215"),
}

# --- MOTOR SPEED ---
# Fixed joint movement step size (in degrees)
STEP_SIZE = 5.0
# Linear mode boundaries (0–100 scale)
LINEAR_MIN_DEGREE = 0.0
LINEAR_MAX_DEGREE = 100.0

# --- KEYBOARD CONTROL ---
# Keyboard key mapping: key to joint and direction
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
# Modifier Keys
KEYBOARD_FAST_MODS = ("shift",)
KEYBOARD_FINE_MODS = ("ctrl", "alt")
# Modifier Multiplier
KEYBOARD_FAST_MULTIPLIER = 3.0
KEYBOARD_FINE_MULTIPLIER = 0.25

# --- PS4 CONTROLLER ---
# DualShock 4 name
PS4_DEVICE_NAME = "Wireless Controller"
# Control loop frequency (Hz)
PS4_LOOP_HZ = 50
# Stick deadzone
PS4_DEADZONE = 0.12
# PS4 Controller key bindings
PS4_BINDINGS = {
    "LX": ("shoulder_pan", +1.0),
    "LY": ("shoulder_lift", -1.0),
    "RY": ("elbow_flex", -1.0),
    "RX": ("wrist_flex", +1.0),
    "L1": ("wrist_roll", -0.6),
    "R1": ("wrist_roll", +0.6),
    "L2": ("gripper", -1.0),
    "R2": ("gripper", +1.0),
}

# --- SERVO CALIBRATION ---
# Load joint limits (deg)
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
# Joint degrees from raw value
def _deg_from_raw(raw_steps: int, offset_steps: int, drive: int, resolution: int) -> float:
    # same math as FeetechBus.read_position
    deg = ( (raw_steps + offset_steps) / (resolution // 2) ) * 180.0
    return -deg if drive else deg
# Load limits from calibration
def _load_limits_from_calibration(calib_path: str) -> Dict[str, Tuple[float, float]]:
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    names      = calib["motor_names"]
    start_pos  = calib["start_pos"]
    end_pos    = calib["end_pos"]
    drive_mode = calib["drive_mode"]
    homing     = calib["homing_offset"]
    # Limits
    limits = {}
    for idx, name in enumerate(names):
        motor_id = SERVOS[name][0]
        offset   = int(homing[str(motor_id)])
        drive    = int(drive_mode[idx])
        a = _deg_from_raw(int(start_pos[idx]), offset, drive, DEFAULT_RESOLUTION)
        b = _deg_from_raw(int(end_pos[idx]), offset, drive, DEFAULT_RESOLUTION)
        lo, hi = (a, b) if a <= b else (b, a)
        limits[name] = (lo, hi)
    return limits
# Load joint limits (deg) with calibration-first policy
try:
    JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = _load_limits_from_calibration(CALIBRATION_PATH)
    if DEBUG:
        print(f"[DEBUG] Joint limits derived from {CALIBRATION_PATH}")
except Exception as ex:
    if DEBUG:
        print(f"[WARN] Could not derive limits from calibration: {ex}\n"
              f"Falling back to {LIMITS_PATH}")
    try:
        JOINT_LIMITS_DEG = _load_joint_limits_from_json(LIMITS_PATH)
        if DEBUG:
            print(f"[DEBUG] Loaded joint limits from {LIMITS_PATH}")
    except Exception as ex2:
        if DEBUG:
            print(f"[WARN] Could not load limits JSON: {ex2}\n Falling back to conservative defaults.")
        JOINT_LIMITS_DEG = {
            "shoulder_pan": (100.0, 290.0),
            "shoulder_lift": (75.0, 288.0),
            "elbow_flex": (78.0, 272.0),
            "wrist_flex": (71.0, 246.0),
            "wrist_roll": (3.6, 342.5),
            "gripper": (183.0, 280.0),
        }

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
    # J1: shoulder_pan (about z). Use α=+90° like before; d=0 now.
    (0.000,       +deg(90.0),  0.000,  deg(MDH_THETA_OFFSETS_DEG["shoulder_pan"])),

    # J2: shoulder_lift (about z in MDH frame; the α on J1 makes this physically a pitch)
    (L1_UPPER_ARM_M, 0.000,    0.000,  deg(MDH_THETA_OFFSETS_DEG["shoulder_lift"])),

    # J3: elbow_flex
    (L2_FOREARM_M,   0.000,    0.000,  deg(MDH_THETA_OFFSETS_DEG["elbow_flex"])),

    # J4: wrist_flex (pitch)
    (L3_WRIST_M,     0.000,    0.000,  deg(MDH_THETA_OFFSETS_DEG["wrist_flex"])),

    # J5: wrist_roll (roll about z)
    (0.000,          0.000,    0.000,  deg(MDH_THETA_OFFSETS_DEG["wrist_roll"])),
]
# Put both shoulder inserts BEFORE joint index 1 (i.e., between J1 and J2)
MDH_FIXED_INSERTS = [
    (1, ("tx", SHOULDER_OFFSET_X_M)),
    (1, ("tz", SHOULDER_OFFSET_Z_M)),
]


# Tool transform
# Rotate tool about its local X/Y/Z (deg)
TOOL_RPY_DEG = (0.0, 0.0, 0.0)
# Offset along tool axis (assume X-axis)
TOOL_POS_M = (TOOL_LENGTH_M, 0.0, 0.0)

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
IK_MAX_ITERS = 5000

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
# Helper: Clamp to joint limits
def clamp_to_limits_deg(q_deg_vec: List[float]) -> List[float]:
    clamped = []
    for q, (lo, hi) in zip(q_deg_vec, limits_for_vector_order()):
        clamped.append(min(max(q, lo), hi))
    return clamped
