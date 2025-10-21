# src/sensors/joint_models.py
import numpy as np
from src.estimation.kf_matrix import bias_density_to_qb, gyro_density_to_var

# Simulate Noisy Encoder (single step)
def simulate_encoder(theta_true, p_dropout, n_ticks, sigma_enc, rng):
    
    # Sample from Uniform distribution in [0, 1)
    bernoulli_sample = rng.random()

    # Dropout probability
    if bernoulli_sample < p_dropout:
        # Return NaN as dropout
        return np.nan
    else: 
        # Sample Gaussian noise
        measurement_noise = rng.normal(0, sigma_enc)
        # Actual measurement
        theta_measurement = theta_true + measurement_noise
        # Theta tick delta
        theta_delta = 2* np.pi / n_ticks 
        # Clipped measurement
        theta_measurement = np.round(theta_measurement / theta_delta) * theta_delta
        # Return measurement
        return theta_measurement

# Simulate Noisy Gyrometer (single step, density inputs)
def simulate_gyro(theta_dot_true, previous_bias, sigma_gyro_density, sigma_drift_density, dt_gyro, rng):
    # Per-step variances from densities
    q_b_t = bias_density_to_qb(sigma_drift_density, dt_gyro)
    sigma_gyro_t_sq = gyro_density_to_var(sigma_gyro_density, dt_gyro)

    # Random-walk bias update
    current_bias = previous_bias + rng.normal(0, np.sqrt(q_b_t))

    # Measurement with white noise
    theta_dot_meas = theta_dot_true + current_bias + rng.normal(0, np.sqrt(sigma_gyro_t_sq))

    return theta_dot_meas, current_bias
