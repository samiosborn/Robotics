# src/utils/quaternion.py
import numpy as np

# Normalise
def normalise(q):
  # Norm
  n = np.linalg.norm(q)
  # Safety
  if n < 1e-12:
    raise ValueError("Cannot normalise zero quaternion")
  # Normalise
  return q / n

# Quaternion multiply
def quat_mul(q1, q2):
  # Unpack
  q1_w, q1_x, q1_y, q1_z = q1
  q2_w, q2_x, q2_y, q2_z = q2
  # Pack
  v1 = np.array([q1_x, q1_y, q1_z])
  v2 = np.array([q2_x, q2_y, q2_z])
  # Scalar
  scal = q1_w * q2_w - np.dot(v1, v2)
  # Vector
  vec = q1_w * v2 + q2_w * v1 + np.cross(v1, v2)
  # Combine
  return np.concatenate([[scal], vec])

# Quaternion conjugate
def quat_conjugate(q):
  # Unpack
  q_w, q_x, q_y, q_z = q
  # Negate the vector components
  return np.array([q_w, -q_x, -q_y, -q_z])

# Quaternion to rotation matrix
def quat_to_rotmat(q):
  # Normalise q
  q = normalise(q)
  # Unpack
  q_w, q_x, q_y, q_z = q
  # Polynomial formula
  return np.array([
    [1-2*(q_y**2+q_z**2), 2*(q_x*q_y-q_z*q_w), 2*(q_x*q_z+q_y*q_w)],
    [2*(q_x*q_y+q_z*q_w), 1-2*(q_x**2+q_z**2), 2*(q_y*q_z-q_x*q_w)],
    [2*(q_x*q_z-q_y*q_w), 2*(q_y*q_z+q_x*q_w), 1-2*(q_x**2+q_y**2)]
  ])

# Rotation vector to quaternion over timestep (exponential map)
def rotvec_to_quat(omega, dt):
  # Rotation vector for timestep
  phi = omega * dt
  # Angle
  theta = np.linalg.norm(phi)
  if theta < 1e-8:
      # Small angle approximation
      w = 1 - 0.125 * theta**2
      v = 0.5 * phi
  else:
      # Axis
      axis = phi / theta
      # Exact angle-axis method
      w = np.cos(theta/2)
      v = axis * np.sin(theta/2)
  return np.concatenate([[w], v])

# Quaternion from axis-angle representation
def quat_from_axis_angle(axis, theta):
  # Combine
  q = np.concatenate([[np.cos(theta/2)], (np.sin(theta/2) * axis)])
  return normalise(q)

# Integrate quaternion over timestep
def integrate_quat(q, omega, dt):
  # Quaternion increment
  dq = rotvec_to_quat(omega, dt)
  # Multiply quaternions
  q_new = quat_mul(dq, q)
  # Normalise
  return normalise(q_new)

# Quaternion log (log mapping)
def quat_log(q):
  # Normalise
  q = normalise(q)
  # Unpack
  q_w, q_x, q_y, q_z = q
  # Pack
  v = np.array([q_x, q_y, q_z])
  # Sin(theta/2)
  sin_half_theta = np.linalg.norm(v)
  if sin_half_theta < 1e-8:
    # Small-angle approximation
    return 2 * v
  else:
    # Theta
    theta = 2 * np.arccos(q_w)
    # Axis
    axis = v / sin_half_theta
    # Axis-angle
    phi = theta * axis
    return phi
