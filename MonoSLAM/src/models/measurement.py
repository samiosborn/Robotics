# src/models/measurement.py
import numpy as np

# Measurement model (Position-only)
def measurement_model(state):
  # h(x) = r
  return state.r

# Measurement Jacobian
def measurement_jacobian():
  # Preallocate
  H = np.zeros((3, 9))
  # Identity
  H[0:3, 0:3] = np.eye(3)
  return H
