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
