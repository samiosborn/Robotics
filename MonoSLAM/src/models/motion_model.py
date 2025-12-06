# src/models/motion_model.py
import numpy as np
from utils.load_config import load_config
from models.state import State
from utils.quaternion import integrate_quat

# Load paths
paths = load_config("src/config/paths.yaml")

# Load constants
constants = load_config(paths["constants_config"])

# Gravity (in world frame)
GRAVITY = np.array(constants["gravity"])

# Propagate state (discrete constant-velocity motion model)
def propagate_state(state: State, omega: np.ndarray, dt: float) -> State:
  # Constant velocity
  r_next = state.r + state.v * dt
  # Velocity update under gravity only
  v_next = state.v + GRAVITY * dt
  # Orientation update
  q_next = integrate_quat(state.q, omega, dt)
  # New state object
  return State(r_next, v_next, q_next)
