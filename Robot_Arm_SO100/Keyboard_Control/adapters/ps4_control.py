# adapters/ps4_control.py
import sys
import time
import threading
from collections import defaultdict
import numpy as np
import evdev
from evdev import ecodes
import config
from control.motion_controller import MotionController

# Apply deadzone
def _apply_deadzone(x: float, dz: float) -> float:
    return 0.0 if abs(x) < dz else x

# Normalise stick
def _norm_stick(v: int, vmin=0, vmax=255, center=128) -> float:
    # map raw 0..255 to -1..1
    return float(np.clip((v - center) / ((vmax - vmin) / 2.0), -1.0, 1.0))

# Normalise trigger
def _norm_trigger(v: int, vmin=0, vmax=255) -> float:
    # map raw 0..255 to 0..1
    return float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))

# Linux evdev reader for PS4 controller (internal class)
class _EvdevPS4Reader:
    def __init__(self, device_name: str):
        # Device
        dev = None
        # Get devices
        for path in evdev.list_devices():
            d = evdev.InputDevice(path)
            if d.name == device_name:
                dev = d
                break
        if dev is None:
            raise RuntimeError(f"PS4 controller '{device_name}' not found via evdev.")
        self.dev = dev
        try:
            # Do not share controller signals
            self.dev.grab()
        except Exception:
            pass

        # State (with defaults)
        self.axes = {"LX": 0.0, "LY": 0.0, "RX": 0.0, "RY": 0.0, "L2": 0.0, "R2": 0.0}
        self.btn = defaultdict(bool)
        self.deadzone = getattr(config, "PS4_DEADZONE", 0.12)
        self.running = True
        self._th = threading.Thread(target=self._reader, daemon=True)

    # Start threading
    def start(self):
        self._th.start()

    # Stop running
    def stop(self):
        self.running = False
        try:
            self.dev.ungrab()
        except Exception:
            pass

    # Read events
    def _reader(self):
        for event in self.dev.read_loop():
            if not self.running:
                break
            # Sticks
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X:
                    self.axes["LX"] = _apply_deadzone(_norm_stick(event.value), self.deadzone)
                elif event.code == ecodes.ABS_Y:
                    self.axes["LY"] = _apply_deadzone(_norm_stick(event.value), self.deadzone)
                elif event.code == ecodes.ABS_RX:
                    self.axes["RX"] = _apply_deadzone(_norm_stick(event.value), self.deadzone)
                elif event.code == ecodes.ABS_RY:
                    self.axes["RY"] = _apply_deadzone(_norm_stick(event.value), self.deadzone)
                elif event.code == ecodes.ABS_Z:
                    self.axes["L2"] = _norm_trigger(event.value)
                elif event.code == ecodes.ABS_RZ:
                    self.axes["R2"] = _norm_trigger(event.value)
            # Triggers
            elif event.type == ecodes.EV_KEY:
                if event.code == ecodes.BTN_TL:
                    self.btn["L1"] = bool(event.value)
                elif event.code == ecodes.BTN_TR:
                    self.btn["R1"] = bool(event.value)
                # Quit
                elif event.code in (ecodes.BTN_START, ecodes.BTN_MODE):
                    if event.value == 1:
                        self.running = False
                        break

# Map current controller state to per-joint deg/s 
def _axes_to_joint_vel(reader: _EvdevPS4Reader) -> dict[str, float]:
    # Joint velocities
    v = {j: 0.0 for j in config.SERVOS.keys()}
    # Key mapping
    b = config.PS4_BINDINGS
    # Max speed per joint
    s = config.MAX_DEG_PER_SEC

    # Sticks
    v[b["LX"][0]] += b["LX"][1] * reader.axes["LX"] * s[b["LX"][0]]
    v[b["LY"][0]] += b["LY"][1] * reader.axes["LY"] * s[b["LY"][0]]
    v[b["RX"][0]] += b["RX"][1] * reader.axes["RX"] * s[b["RX"][0]]
    v[b["RY"][0]] += b["RY"][1] * reader.axes["RY"] * s[b["RY"][0]]

    # Triggers
    v[b["L2"][0]] += b["L2"][1] * reader.axes["L2"] * s[b["L2"][0]]
    v[b["R2"][0]] += b["R2"][1] * reader.axes["R2"] * s[b["R2"][0]]

    # Bumpers (constant-rate when held)
    if reader.btn["L1"]:
        v[b["L1"][0]] += b["L1"][1] * s[b["L1"][0]]
    if reader.btn["R1"]:
        v[b["R1"][0]] += b["R1"][1] * s[b["R1"][0]]

    return v

# Public function to run teleop via PS4 controller
def run_ps4_teleop():
    # Import device and begin reading
    reader = _EvdevPS4Reader(config.PS4_DEVICE_NAME)
    reader.start()
    # Time step
    dt = 1.0 / max(1, getattr(config, "PS4_LOOP_HZ", 50))
    # Import motion controller
    ctrl = MotionController()

    print("PS4 teleop running. Hold OPTIONS/PS to quit.")
    try:
        while reader.running:
            # Convert controller reading into: dict[joint] of deg/s
            vel = _axes_to_joint_vel(reader)
            # Apply joint velocity to return new position at end of time step
            ctrl.apply_velocities(vel, dt)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        ctrl.close()
        if config.DEBUG:
            print("[INFO] PS4 teleop stopped.")


if __name__ == "__main__":
    run_ps4_teleop()
