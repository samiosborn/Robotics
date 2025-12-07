# src/ekf/prediction.py
from ekf.covariance import process_noise_covariance
from models.motion_model import propagate_state
from models.jacobians import state_transition_jacobian

def ekf_predict(state, P, omega, dt, ekf_cfg):
  # Process noise
  sigma_r = ekf_cfg["process_noise_std"]["position"]
  sigma_v = ekf_cfg["process_noise_std"]["velocity"]
  sigma_omega = ekf_cfg["process_noise_std"]["orientation"]
  # Process covariance
  Q = process_noise_covariance(sigma_r, sigma_v, sigma_omega, dt)
  # Predicted state
  state_pred = propagate_state(state, omega, dt)
  # State transition jacobian
  F = state_transition_jacobian(omega, dt)
  # Predicted covariance
  P_pred = F @ P @ F.T + Q
  return state_pred, P_pred
