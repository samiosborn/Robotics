# src/dynamics/spatial_math.py
import numpy as np

# Convert 3-vector into skew-symmetric matrix (3x3) - so that skew3(a) @ b = a x b
def skew3(v):
   x, y, z = v
   return np.array([
      [0, -z, y],
      [z, 0, -x],
      [-y, x, 0]
   ])

# Convert skew-symmetric matrix (3x3) into 3-vector
def vec3_from_skew(S):
    return np.array([-S[1,2], S[0,2], -S[0,1]], dtype=float)
