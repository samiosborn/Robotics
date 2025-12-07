# src/models/jacobians.py
import numpy as np
from utils.linalg import hat

# Jacobian for state transition matrix (constant-velocity with gravity-only motion model)
def state_transition_jacobian(omega: np.ndarray, dt: float) -> np.ndarray:
  # Pre-allocate F
  F = np.zeros((9, 9))
  # Partial derivatives of position
  F[0:3, 0:3] = np.eye(3)
  F[0:3, 3:6] = dt * np.eye(3)
  F[0:3, 6:9] = np.zeros((3, 3))
  # Partial derivatives of velocity
  F[3:6, 0:3] = np.zeros((3, 3))
  F[3:6, 3:6] = np.eye(3)
  F[3:6, 6:9] = np.zeros((3, 3))
  # Partial derivatives of orientation
  F[6:9, 0:3] = np.zeros((3, 3))
  F[6:9, 3:6] = np.zeros((3, 3))
  F[6:9, 6:9] = np.eye(3) - dt * hat(omega)
  return F
