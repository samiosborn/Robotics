# src/robot/control/balance_controller.py
import yaml

class BalanceController: 
    def __init__(self, control_yaml_config_path, pid_object):
        # Import YAML Config
        with open(control_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)["balance"]

        # --- Class Variables ---
        # Deadband for balancing
        self._deadband = cfg["deadband"]
        # Angle offset (radians)
        self._angle_offset = cfg["angle_offset"]
        # PID instance
        self._pid = pid_object

    # --- Private Methods ---
    # Apply Angle Offset
    def _apply_angle_offset(self, theta):
        theta_corrected = theta - self._angle_offset
        return theta_corrected
    # Apply deadband
    def _apply_deadband(self, duty):
        if abs(duty) < self._deadband:
            return 0.0
        else: 
            return duty
    
    # --- Public API ---
    # Update
    def update(self, theta, theta_dot, dt):
        # Angle offset
        theta_error = self._apply_angle_offset(theta)
        # PID duty
        duty = self._pid.update(theta_error, theta_dot, dt)
        # Apply deadband
        left_duty = self._apply_deadband(duty)
        right_duty = self._apply_deadband(duty)
        return left_duty, right_duty
