# src/robot/imu/complementary_filter.py
import yaml
import math

class ComplementaryFilter():
    def __init__(self, control_yaml_config_path):
        # Load YAML config
        with open(control_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)["filter"]

        # --- Class Variables ---
        # Fusion coefficient alpha
        self._alpha = cfg["alpha"]

        # --- Instance Variables ---
        # Gyro pitch rate 
        self._pitch_rate = 0.0
        # Pitch tilt angle estimate
        self._theta_est = 0.0
        # For first update of theta_est
        self._initialised = False

    # --- Public API ---
    # Expose theta
    @property
    def theta(self):
        return self._theta_est
    # Expose theta_dot
    @property
    def theta_dot(self):
        return self._pitch_rate
    # Reset the estimated angle (to zero)
    def reset(self, theta0=0.0):
        self._theta_est = theta0
        self._initialised = True
    # Update the estimate
    def update(self, accel_reading, gyro_reading, dt):
        # Theta from accel
        theta_acc = math.atan2(-accel_reading["roll"], accel_reading["yaw"])
        # Initialise from accelerometer
        if not self._initialised:
            self._theta_est = theta_acc
            self._initialised = True
            return self._theta_est
        # Gyro pitch rate (rad/s)
        self._pitch_rate = gyro_reading["pitch"]
        # Theta from gyro
        theta_gyro = self._theta_est + self._pitch_rate * dt
        # Complementary filter
        self._theta_est = self._alpha * theta_gyro + (1 - self._alpha) * theta_acc
        return self._theta_est
