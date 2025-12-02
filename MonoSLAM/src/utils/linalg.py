# src/utils/linalg.py
import numpy as np

# Hat operator (skew-symmetric matrix)
def hat(omega):
  # Unpack
  w_x, w_y, w_z = omega
  return np.array([
    [0, -w_z, w_y],
    [w_z, 0, -w_x],
    [-w_y, w_x, 0]
  ], dtype = float)

# Vee (inverse of hat operator)
def vee(W):
  return np.array([W[2,1], W[0,2], W[1,0]])

# Small-angle approximation for SO(3) exponential
def so3_exp_small(omega, dt):
  # R = I + omega x dt
  return np.eye(3) + hat(omega) * dt
