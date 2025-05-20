# keyboard_control.py
import keyboard
import time
import json
import os
import config
from feetech_bus import FeetechBus

# Check if any control key is pressed
def get_active_command():
    for key, command in config.KEYBOARD_BINDINGS.items():
        if keyboard.is_pressed(key):
            return command
    return None

# Load the most recent calibration file in the expected runtime format
def load_latest_calibration() -> dict:
    folder = config.CALIBRATION_FOLDER
    if not os.path.exists(folder):
        raise FileNotFoundError("No configs directory found.")

    files = [f for f in os.listdir(folder) if f.startswith("calib_") and f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No calibration files found in configs/")

    files.sort(reverse=True)
    latest = files[0]
    path = os.path.join(folder, latest)

    with open(path, "r") as f:
        calibration = json.load(f)

    if config.DEBUG:
        print(f"[DEBUG] Loaded latest calibration file: {path}")

    return calibration

# Initialise the Feetech bus and apply calibration
def create_motor_bus(calib_path: str = config.CALIBRATION_PATH) -> FeetechBus:
    if config.DEBUG:
        print(f"[DEBUG] Initialising FeetechBus on {config.SERVO_PORT} with motors: {config.SERVOS}")

    calibration = load_latest_calibration()

    bus = FeetechBus(port=config.SERVO_PORT, motors=config.SERVOS)
    bus.connect()
    bus.set_calibration(calibration)

    if config.DEBUG:
        print("[DEBUG] Calibration applied to motor bus")

    return bus

# Global motor bus and joint angle cache
_motor_bus = None
_joint_angles = {}

# Execute a single teleoperation step for the specified joint and direction
def execute_teleop_command(joint: str, direction: int) -> tuple[float, float]:
    global _motor_bus, _joint_angles

    if _motor_bus is None:
        _motor_bus = create_motor_bus()
        for name in config.SERVOS:
            _joint_angles[name] = _motor_bus.read_position(name)

    idx = _motor_bus.calibration["motor_names"].index(joint)
    mode = _motor_bus.calibration["calib_mode"][idx]
    start = _motor_bus.calibration["start_pos"][idx]
    end = _motor_bus.calibration["end_pos"][idx]
    motor_id = config.SERVOS[joint][0]
    offset = _motor_bus.calibration["homing_offset"][str(motor_id)]
    drive = _motor_bus.calibration["drive_mode"][idx]
    resolution = config.DEFAULT_RESOLUTION

    linear_min_deg = config.LINEAR_MIN_DEGREE
    linear_max_deg = config.LINEAR_MAX_DEGREE

    if mode == "DEGREE":
        deg_min = (start + offset) / (resolution // 2) * 180
        deg_max = (end + offset) / (resolution // 2) * 180
        if drive:
            deg_min, deg_max = -deg_max, -deg_min
    else:
        deg_min, deg_max = linear_min_deg, linear_max_deg

    deg_min, deg_max = sorted([deg_min, deg_max])

    before = _joint_angles[joint]
    proposed = before + direction * config.STEP_SIZE

    if proposed < deg_min or proposed > deg_max:
        if config.DEBUG:
            print(f"[DEBUG] {joint} angle {proposed:.1f}° exceeds limits ({deg_min:.1f}° to {deg_max:.1f}°) → clamped.")
        return before, before

    moved = _motor_bus.write_position(joint, proposed)
    if moved:
        _joint_angles[joint] = proposed
        if config.DEBUG:
            print(f"[DEBUG] {joint} {direction:+d} -> {before:.1f}° → {proposed:.1f}°")
    else:
        if config.DEBUG:
            print(f"[DEBUG] {joint} at limit ({before:.1f}°), no movement.")

    return before, _joint_angles[joint]

# Run the main keyboard-controlled teleoperation loop
def main_loop():
    print("Teleop running. Press 'z' to quit.")

    try:
        while True:
            cmd = get_active_command()
            if cmd is None:
                continue

            if cmd == "quit":
                raise KeyboardInterrupt

            joint, direction = cmd
            before, after = execute_teleop_command(joint, direction)

            if config.DEBUG:
                print(f"[DEBUG] {joint} {direction:+d} -> {before:.1f}° → {after:.1f}°")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Teleop stopped by user")
        if _motor_bus:
            _motor_bus.disconnect()

if __name__ == "__main__":
    main_loop()
