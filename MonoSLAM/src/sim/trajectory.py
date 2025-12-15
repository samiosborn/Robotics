# src/sim/trajectory.py
import numpy as np
from utils.quaternion import quat_from_axis_angle

# Generate anticlockwise-circular trajectory (xy-plane)
def generate_circular_traj(R, omega_z, T, dt):

  # Pre-allocate
  t = np.arange(0, T, dt)
  N = len(t)
  positions = np.zeros((N, 3))
  velocities = np.zeros((N, 3))
  orientations = np.zeros((N, 4))

  # Loop over all timesteps
  for i, ti in enumerate(t):
    # Positions
    x = R * np.cos(omega_z * ti)
    y = R * np.sin(omega_z * ti)
    z = 0.0
    positions[i] = np.array([x, y, z])

    # Velocities
    v_x = -R * omega_z * np.sin(omega_z * ti)
    v_y = R * omega_z * np.cos(omega_z * ti)
    v_z = 0.0
    velocities[i] = np.array([v_x, v_y, v_z])

    # Orientation
    yaw_angle = np.arctan2(y, x)
    z_rotation_axis = np.array([0, 0, 1])
    orientations[i] = quat_from_axis_angle(z_rotation_axis, yaw_angle)

  return positions, velocities, orientations
