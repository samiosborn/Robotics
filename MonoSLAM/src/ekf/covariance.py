# src/ekf/covariance.py
import numpy as np

# Process noise covariance
def process_noise_covariance(sigma_r, sigma_v, sigma_omega, dt):
  # Pre-allocate Q
  Q = np.zeros((9, 9))
  # Position noise
  Q[0:3, 0:3] = (sigma_r**2) * np.eye(3) * dt
  # Velocity noise
  Q[3:6, 3:6] = (sigma_v**2) * np.eye(3) * dt
  # Orientation noise
  Q[6:9, 6:9] = (sigma_omega**2) * np.eye(3) * dt
  return Q
