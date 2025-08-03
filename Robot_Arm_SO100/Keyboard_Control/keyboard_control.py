# keyboard_control.py
import keyboard
import time
import json
import os
import config
from feetech_bus import FeetechBus

# Check if any control key is pressed
def get_active_command():
    # Get keyboard inputsq
    for key, command in config.KEYBOARD_BINDINGS.items():
        if keyboard.is_pressed(key):
            return command
    return None

# Initialise the Feetech bus and apply calibration
def create_motor_bus(calib_folder: str = config.CALIBRATION_FOLDER, calib_path: str = config.CALIBRATION_PATH) -> FeetechBus:
    import os
    import json

    # Full path using folder and calib file name
    full_path = os.path.join(calib_folder, calib_path)

    if config.DEBUG:
        print(f"[DEBUG] Initialising FeetechBus on {config.SERVO_PORT} with motors: {config.SERVOS}")
        print(f"[DEBUG] Loading calibration from: {full_path}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"[ERROR] Calibration file not found at '{full_path}'")

    with open(full_path, "r") as f:
        calibration = json.load(f)

    # Create servo bus and set calibration
    bus = FeetechBus(port=config.SERVO_PORT, motors=config.SERVOS)
    bus.connect()
    bus.set_calibration(calibration)

    if config.DEBUG:
        print("[DEBUG] Calibration applied to motor bus")

    return bus

# Global motor bus and joint angle cache
_motor_bus = None
_joint_angles = {}

# Execute a single teleop command for a single joint
def execute_teleop_command(joint: str, direction: int) -> tuple[float, float]:
    global _motor_bus, _joint_angles

    # Create motor bus
    if _motor_bus is None:
        _motor_bus = create_motor_bus()
        for name in config.SERVOS:
            # Read positions
            _joint_angles[name] = _motor_bus.read_position(name)

    # Set calibration parameters
    idx = _motor_bus.calibration["motor_names"].index(joint)
    mode = _motor_bus.calibration["calib_mode"][idx]
    start = _motor_bus.calibration["start_pos"][idx]
    end = _motor_bus.calibration["end_pos"][idx]
    motor_id = config.SERVOS[joint][0]
    offset = _motor_bus.calibration["homing_offset"][str(motor_id)]
    drive = _motor_bus.calibration["drive_mode"][idx]
    resolution = config.DEFAULT_RESOLUTION

    # Linear mode (unused)
    linear_min_deg = config.LINEAR_MIN_DEGREE
    linear_max_deg = config.LINEAR_MAX_DEGREE

    # Degree mode: convert raw values to angles
    if mode == "DEGREE":
        deg_min = (start + offset) / (resolution // 2) * 180
        deg_max = (end + offset) / (resolution // 2) * 180
        # Flip directions if backwards (drive reversal)
        if drive:
            deg_min, deg_max = -deg_max, -deg_min
    else:
        deg_min, deg_max = linear_min_deg, linear_max_deg

    deg_min, deg_max = sorted([deg_min, deg_max])

    before = _joint_angles[joint]

    # Target joint angle
    proposed = before + direction * config.STEP_SIZE

    # Check if within limits
    if proposed < deg_min or proposed > deg_max:
        if config.DEBUG:
            print(f"[DEBUG] {joint} angle {proposed:.1f}° exceeds limits ({deg_min:.1f}° to {deg_max:.1f}°) → clamped.")
        return before, before

    # Move to position
    moved = _motor_bus.write_position(joint, proposed)

    # Print debug statements about new position
    if moved:
        _joint_angles[joint] = proposed
        if config.DEBUG:
            print(f"[DEBUG] {joint} {direction:+d} -> {before:.1f}° → {proposed:.1f}°")
    else:
        if config.DEBUG:
            print(f"[DEBUG] {joint} at limit ({before:.1f}°), no movement.")

    return before, _joint_angles[joint]

# Run the main teleop loop
def main_loop():
    print("Teleop running. Press 'z' to quit. Click away from terminal and code. ")

    try:
        while True:
            # Get keyboard commands
            cmd = get_active_command()
            if cmd is None:
                continue
            
            # Quit
            if cmd == "quit":
                raise KeyboardInterrupt

            # Get command and execute
            joint, direction = cmd
            before, after = execute_teleop_command(joint, direction)

            if config.DEBUG:
                print(f"[DEBUG] {joint} {direction:+d} -> {before:.1f}° → {after:.1f}°")

            # Brief pause to apply
            time.sleep(0.05)

    except KeyboardInterrupt:
        # Quit and disconnect
        print("\n[INFO] Teleop stopped by user")
        if _motor_bus:
            _motor_bus.disconnect()

if __name__ == "__main__":
    main_loop()
