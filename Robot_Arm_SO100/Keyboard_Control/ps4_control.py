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
        self.maxspd   = config.MAX_DEG_PER_SEC

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

    # Create and calibrate the motor bus
    def _create_motor_bus(self) -> FeetechBus:
        # Import for path and JSON handling
        import os, json
        # Build absolute path to the calibration file
        full_path = os.path.join(config.CALIBRATION_FOLDER, config.CALIBRATION_PATH)
        # Ensure the calibration file exists
        if not os.path.exists(full_path):
            # Raise a clear error if missing
            raise FileNotFoundError(f"Calibration file not found at '{full_path}'")
        # Open and read the file
        with open(full_path, "r") as f:
            # Parse calibration JSON
            calibration = json.load(f)
        # Instantiate bus with configured serial port and servo map
        bus = FeetechBus(port=config.SERVO_PORT, motors=config.SERVOS)
        # Open serial connection
        bus.connect()
        # Apply calibration so limits/drive are respected
        bus.set_calibration(calibration)
        # Return ready-to-use bus
        return bus

    # Background event reader thread
    def _event_reader(self):
        # Loop over events from the controller
        for event in self.dev.read_loop():
            # Handle absolute axis events (sticks/triggers)
            if event.type == ecodes.EV_ABS:
                # Left stick X
                if event.code == ecodes.ABS_X:
                    # Normalise and apply deadzone
                    self.axes["LX"] = apply_deadzone(norm_stick(event.value), self.deadzone)
                # Left stick Y
                elif event.code == ecodes.ABS_Y:
                    self.axes["LY"] = apply_deadzone(norm_stick(event.value), self.deadzone)
                # Right stick X
                elif event.code == ecodes.ABS_RX:
                    self.axes["RX"] = apply_deadzone(norm_stick(event.value), self.deadzone)
                # Right stick Y
                elif event.code == ecodes.ABS_RY:
                    self.axes["RY"] = apply_deadzone(norm_stick(event.value), self.deadzone)
                # Left trigger (0..1)
                elif event.code == ecodes.ABS_Z:
                    # Normalise to 0..1
                    self.axes["L2"] = normalise_trigger(event.value)
                # Right trigger
                elif event.code == ecodes.ABS_RZ:
                    self.axes["R2"] = normalise_trigger(event.value)

            # Handle button events
            elif event.type == ecodes.EV_KEY:
                # L1 pressed/released
                if event.code == ecodes.BTN_TL:
                    # Store button state
                    self.btn["L1"] = bool(event.value)
                # R1 pressed/released
                elif event.code == ecodes.BTN_TR:
                    self.btn["R1"] = bool(event.value)
                # Options or PS button to quit
                elif event.code in (ecodes.BTN_START, ecodes.BTN_MODE):
                    # On press, request stop and break loop
                    if event.value == 1:
                        self.running = False
                        break
        return

    # Convert current axes/buttons to joint velocities
    def _axis_to_joint_vel(self) -> dict:
        # Start with zero velocity for each joint
        v = {jn: 0.0 for jn in config.SERVOS.keys()}
        # Alias
        b = self.bindings
        s = self.maxspd

        # Map left stick X to joint velocity
        v[b["LX"][0]] += b["LX"][1] * self.axes["LX"] * s[b["LX"][0]]
        # Map left stick Y to joint velocity
        v[b["LY"][0]] += b["LY"][1] * self.axes["LY"] * s[b["LY"][0]]
        # Map right stick X to joint velocity
        v[b["RX"][0]] += b["RX"][1] * self.axes["RX"] * s[b["RX"][0]]
        # Map right stick Y to joint velocity
        v[b["RY"][0]] += b["RY"][1] * self.axes["RY"] * s[b["RY"][0]]

        # Map left trigger to joint velocity (0..1)
        v[b["L2"][0]] += b["L2"][1] * self.axes["L2"] * s[b["L2"][0]]
        # Map right trigger to joint velocity (0..1)
        v[b["R2"][0]] += b["R2"][1] * self.axes["R2"] * s[b["R2"][0]]

        # If L1 is held, add constant roll rate
        if self.btn["L1"]:
            v[b["L1"][0]] += b["L1"][1] * s[b["L1"][0]]
        # If R1 is held, add constant roll rate
        if self.btn["R1"]:
            v[b["R1"][0]] += b["R1"][1] * s[b["R1"][0]]

        # Return per-joint velocities in deg/s
        return v

    # Main control loop
    def run(self):
        # Announce start
        print("PS4 teleop running. Press OPTIONS/PS to quit.")
        # Start background event reader
        self.reader_th.start()
        # Loop until stop requested
        try:
            # While running flag is true
            while self.running:
                # Compute desired joint velocities from inputs
                vel = self._axis_to_joint_vel()
                # For each joint velocity
                for joint, w in vel.items():
                    # Skip near-zero commands
                    if abs(w) < 1e-6:
                        continue
                    # Read last cached angle
                    before = self.joint_angles[joint]
                    # Integrate velocity over fixed timestep
                    target = before + w * self.dt
                    # Command the servo; bus enforces limits
                    moved = self.bus.write_position(joint, target)
                    # If movement accepted, update cache
                    if moved:
                        self.joint_angles[joint] = target
                # Sleep to maintain loop rate
                time.sleep(self.dt)
        # Ensure cleanup on any exception or normal exit
        finally:
            self.running = False
            # Try to ungrab device (ignore errors)
            try:
                self.dev.ungrab()
            except Exception:
                pass
            # Disconnect servo bus
            self.bus.disconnect()
            if config.DEBUG:
                print("[INFO] PS4 teleop stopped.")

# Entry
if __name__ == "__main__":
    # Create teleop object
    teleop = PS4Teleop()
    # Run the control loop
    teleop.run()
