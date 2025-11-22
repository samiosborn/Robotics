# src/robot/control/pid.py
import yaml

class PID(): 
    def __init__(self, control_yaml_config_path):
        # Import YAML config
        with open(control_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)

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
        # Derivative (previous)
        self._prev_derivative = 0.0
        # Integral
        self._integral = 0.0

    # --- Private Methods ---
    # Clamp to liitss
    def _clamp(self, value, limits):
        return max(limits["min"], min(value, limits["max"]))

    # --- Public API ---
    # Reset
    def reset(self):
        # Error (previous)
        self._prev_error = 0.0
        # Derivative (previous)
        self._prev_derivative = 0.0
        # Integral
        self._integral = 0.0
    # Update
    def update(self, error, dt):
        # Proportional component
        P = error * self._kp
        # Integral 
        self._integral += error * dt
        # Anti-windup clamp
        self._integral = self._clamp(self._integral, self._integral_limits)
        # Integral component
        I = self._integral * self._ki
        # Filtered derivative
        if dt <= 0:
            dt = 1e-6
        raw_derivative = (error - self._prev_error) / dt
        filtered_derivative = self._beta * self._prev_derivative + (1 - self._beta) * raw_derivative
        # Derivative component
        D = filtered_derivative * self._kd
        # Store previous terms
        self._prev_error = error
        self._prev_derivative = filtered_derivative
        # Combine
        u = P + I + D
        # Clip output
        u = self._clamp(u, self._output_limits)
        return u
