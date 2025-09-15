# ps4_control.py

import time
import math
import threading
from collections import defaultdict
import evdev
from evdev import ecodes
import config
from feetech_bus import FeetechBus

# Find PS4 Device
def find_ps4_device(name = config.PS4_DEVICE_NAME):
    paths = evdev.list_devices()
    # Loop across all paths
    for p in paths: 
        # Connect to this device
        dev = evdev.InputDevice(p)
        # Check it's the right name
        if dev.name == name: 
            return dev
    # No relevant device
    return None

# --- NORMALISE DUALSHOCK 4 AXES ---
# Normalise stick to [-1.0, 1.0]
def norm_stick(v, vmin = 0, vmax = 255, center = 128):
    # Center and normalise
    x = (v - center) / ((vmax - vmin) / 2.0)
    # Clamp and return
    return max(-1.0, min(1.0, x))

# Normalise trigger to [0.0, 1.0]
def normalise_trigger(v, vmin = 0, vmax = 255):
    # Center and normalise
    x = (v - vmin) / (vmax - vmin)
    # Clamp and return
    return max(0.0, min(1.0, x))

# Apply deadzone (send to 0 within deadzone)
def apply_deadzone(x, dz):
    if abs(x) < dz: 
        return 0.0
    else:
        return x

# --- PS4 TELEOP CLASS ---
class PS4Teleop: 
    def __init__(self):
        # Find and grab the controller
        self.dev = find_ps4_device()
        if self.dev is None:
            raise RuntimeError(f"PS4 controller '{config.PS4_DEVICE_NAME}' not found.")
        try:
            # Exclusive capture
            self.dev.grab()
        except Exception:
            pass

        # Normalised axis state
        self.axes = {"LX": 0.0, "LY": 0.0, "RX": 0.0, "RY": 0.0, "L2": 0.0, "R2": 0.0}
        # Button state
        self.btn = defaultdict(bool)

        # Motion mappings and speed limits from config
        self.bindings = config.PS4_BINDINGS
        self.maxspd   = config.PS4_MAX_DEG_PER_SEC

        # Servo bus with calibration
        self.bus = self._create_motor_bus()

        # Joint angle cache (start from current positions)
        self.joint_angles = {name: self.bus.read_position(name) for name in config.SERVOS}

        # Control loop timing
        self.loop_hz = getattr(config, "PS4_LOOP_HZ", 50)
        self.dt = 1.0 / max(1, self.loop_hz)

        # Deadzone for sticks
        self.deadzone = getattr(config, "PS4_DEADZONE", 0.15)

        # Run state flag
        self.running = True

        # Background event reader (started in run())
        self.reader_th = threading.Thread(target=self._event_reader, daemon=True)
