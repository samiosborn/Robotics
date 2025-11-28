# src/robot/control/pid.py
import yaml

class PID(): 
    def __init__(self, control_yaml_config_path):
        # Import YAML config
        with open(control_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)["pid"]

        # --- Class Variables ---
        # Derivative low-pass filter coefficient
        self._beta = cfg["beta"]
        # Proportional gain
        self._kp = cfg["kp"]
        # Integral gain
        self._ki = cfg["ki"]
        # Derivative gain
        self._kd = cfg["kd"]
        # Intregral limits
        self._integral_limits = cfg["integral_limits"]
        # Output limits
        self._output_limits = cfg["output_limits"]

        # --- Instance Variables ---
        # Error (previous)
        self._prev_error = 0.0
        # Integral
        self._integral = 0.0

    # --- Private Methods ---
    # Clamp to limits
    def _clamp(self, value, limits):
        return max(limits["min"], min(value, limits["max"]))

    # --- Public API ---
    # Reset
    def reset(self):
        # Error (previous)
        self._prev_error = 0.0
        # Integral
        self._integral = 0.0
    # Update
    def update(self, theta_error, theta_dot, dt):
        # Proportional component
        P = theta_error * self._kp
        # Integral 
        self._integral += theta_error * dt
        # Anti-windup clamp
        self._integral = self._clamp(self._integral, self._integral_limits)
        # Integral component
        I = self._integral * self._ki
        # Derivative component
        D = -theta_dot * self._kd
        # Store previous terms
        self._prev_error = theta_error
        # Combine
        u = P + I + D
        # Clip output
        u = self._clamp(u, self._output_limits)
        return u
