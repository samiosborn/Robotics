# control/motion_controller.py
import os
import json
import time
from typing import Callable, Dict, Tuple, Optional
import config
from control.feetech_bus import FeetechBus

# Command per step (joint, delta_deg)
Command = Tuple[str, float]
# Velocity commands as a dict of deg/s per joint
Velocities = Dict[str, float]

# Group motor control
class MotionController:
    def __init__(self):
        # Connect with the motor bus
        self.bus = self._create_motor_bus()
        # Cache of recent joint angles
        self.joint_angles = {name: self.bus.read_position(name) for name in config.SERVOS}

    # Create motor bus
    def _create_motor_bus(self) -> FeetechBus:
        # Absolute path to the calibration file
        full_path = os.path.abspath(config.CALIBRATION_PATH)
        # Ensure calibration file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Calibration file not found at '{full_path}'")
        # Open the calibration file
        with open(full_path, "r") as f:
            # Read the JSON into a dictionary
            calibration = json.load(f)
        # Instantiate the bus using the serial port and servo motors mapping
        bus = FeetechBus(port=config.SERVO_PORT, motors=config.SERVOS)
        # Open the serial link to the servo chain
        bus.connect()
        # Push the calibration data into the bus so it can enforce limits
        bus.set_calibration(calibration)
        # Return the ready-to-use bus
        return bus

    # Apply a single step for one joint
    def apply_step(self, joint: str, step_deg: float) -> Tuple[float, float]:
        # Read the last known angle for the selected joint
        before = self.joint_angles[joint]
        # Compute the desired new target
        target = before + step_deg
        # Command the bus to move to the target position
        moved = self.bus.write_position(joint, target)
        # commentary: if the move happened, update our cache and after-value
        if moved:
            # Store the new joint angle locally
            self.joint_angles[joint] = target
            # Update after position
            after = target
        else: 
            # Move was rejected, update after (to before)
            after = before
        # Print debug (before and after and step)
        if config.DEBUG:
            print(f"[STEP] {joint}: {before:.1f} to {after:.1f} (delta {step_deg:+.1f})")
        # Return both before and after for logging (optional)
        return before, after

    # Apply continuous velocity commands
    def apply_velocities(self, vel_deg_s: Velocities, dt: float) -> None:
        # Iterate over each joint
        for joint, w in vel_deg_s.items():
            # Skip no-ops
            if abs(w) < 1e-6:
                continue
            # Read the last angle
            before = self.joint_angles[joint]
            # Integrate velocity over timestep to get the next target
            target = before + w * dt
            # Write position to servo
            moved = self.bus.write_position(joint, target)
            if moved:
                # Update joint angle tracker
                self.joint_angles[joint] = target
                # Debug (positions and commanded speed)
                if config.DEBUG:
                    print(f"[VEL] {joint}: {before:.1f} to {target:.1f} @ {w:+.1f} deg/s")

    # Close bus connection
    def close(self):
        try:
            # Close the serial port to the servo bus
            self.bus.disconnect()
        # Ignore any exceptions
        except Exception:
            pass

# Run a polling loop for discrete step-based inputs (e.g., keyboard)
def run_step_loop(get_command: Callable[[], Optional[Command]], step_sleep: float = 0.05):
    # Create a motor controller instance
    ctrl = MotionController()
    # Let the user know how to exit
    print("Step teleop running. Press the adapter's quit key to exit.")
    try:
        while True:
            # Fetch a command tuple
            cmd = get_command()
            if cmd is None:
                # Pause briefly before polling again
                time.sleep(step_sleep)
                continue
            # Request a clean exit
            if cmd == ("__QUIT__", 0.0):
                break
            # Unpack the joint name and signed step size
            joint, ddeg = cmd
            # Apply step
            ctrl.apply_step(joint, ddeg)
            # Small sleep to pace command rate
            time.sleep(step_sleep)
    finally:
        # Disconnect from the bus
        ctrl.close()
        if config.DEBUG:
            print("[INFO] Step teleop stopped.")

# Run a fixed-rate loop for continuous velocity inputs (e.g., PS4 stick controller)
def run_velocity_loop(get_velocities: Callable[[], Velocities], hz: int):
    # Clamp at 1 Hz minimum
    dt = 1.0 / max(1, hz)
    # Construct the controller
    ctrl = MotionController()
    # Print the operating frequency and exit reminder
    if config.DEBUG:
        print(f"Velocity teleop running at {hz} Hz. Use the adapter's quit to exit.")
    try:
        while True:
            # Get current joint velocities
            vels = get_velocities()
            # Apply velocities over the fixed timestep
            ctrl.apply_velocities(vels, dt)
            # Sleep to maintain the target loop frequency
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.close()
        if config.DEBUG:
            print("[INFO] Velocity teleop stopped.")
