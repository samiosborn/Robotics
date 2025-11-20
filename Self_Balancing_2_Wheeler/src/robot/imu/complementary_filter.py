# src/robot/imu/complementary_filter.py
import yaml
import math

class ComplementaryFilter():
    def __init__(self, control_yaml_config_path):
        # Load YAML config
        with open(control_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)

        # --- Class Variables ---
        # Fusion coefficient alpha
        self._alpha = cfg["alpha"]

        # --- Instance Variables ---
        # Pitch tilt angle estimate
        self._theta_est = 0.0
        # For first update of theta_est
        self._initialised = False

    # --- Public API ---
    # Expose theta
    @property
    def theta(self):
        return self._theta_est
    # Reset the estimated angle (to zero)
    def reset(self, theta0=0.0):
        self._theta_est = theta0
        self._initialised = True
    # Update the estimate
    def update(self, accel, gyro, dt):
        # Theta from accel
        theta_acc = math.atan2(-accel["roll"], accel["yaw"])
        # Initialise from accelerometer
        if not self._initialised:
            self._theta_est = theta_acc
            self._initialised = True
            return self._theta_est
        # Gyro pitch rate (rad/s)
        pitch_rate = gyro["pitch"]
        # Theta from gyro
        theta_gyro = self._theta_est + pitch_rate * dt
        # Complementary filter
        self._theta_est = self._alpha * theta_gyro + (1 - self._alpha) * theta_acc
        return self._theta_est
