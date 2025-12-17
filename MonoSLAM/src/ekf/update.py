# src/ekf/update.py
import numpy as np
from models.measurement import measurement_model, measurement_jacobian
from utils.quaternion import rotvec_to_quat, quat_mul, normalise
from models.state import State

# EKF Update step
def ekf_update(state_pred, P_pred, z, R):
  # Innovation (residual)
  z_pred = measurement_model(state_pred)
  y = z - z_pred

  # Jacobian
  H = measurement_jacobian()

  # Innovation covariance
  S = H @ P_pred @ H.T + R

  # Kalman gain
  K = P_pred @ H.T @ np.linalg.inv(S)

  # Covariance update
  I = np.eye(9)
  P_update = (I - K @ H) @ P_pred

  # Error-state correction
  delta_x = K @ y
  delta_r = delta_x[0:3]
  delta_v = delta_x[3:6]
  delta_theta = delta_x[6:9]

  # Apply correction to position and velocity
  r_update = state_pred.r + delta_r
  v_update = state_pred.v + delta_v

  # Correct pose
  dq = rotvec_to_quat(delta_theta, 1.0)
  q_update = quat_mul(dq, state_pred.q)
  q_update = normalise(q_update)

  # Updated state
  state_update = State(r_update, v_update, q_update)

  return state_update, P_update
