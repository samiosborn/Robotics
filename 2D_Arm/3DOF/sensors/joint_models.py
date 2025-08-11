# joint_models.py
import numpy as np

# Number of ticks per revolution
n_ticks = 4096

# Probability of dropout
p_dropout = 0.05

# Encoder observation noise standard deviation
sigma_enc = 1.0

# Encoder time step
dt_enc = 0.1

# Random Number Generator (with magic number = 42)
rng = np.random.default_rng(seed=42)

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

# Gyro observation noise standard deviation
sigma_gyro = 1.0

# Gyro previous step bias observation noise standard deviation
sigma_previous_bias = 1.0

# Gyro drift observation noise standard deviation
sigma_drift = 1.0

# Previous step bias
previous_bias = 1.0

# Real angular velocity
theta_dot_true = 1.0

# Gyro time step
dt_gyro = 0.025

# Simulate Noisy Gyrometer (single step, density inputs)
def simulate_gyro(theta_dot_true, previous_bias, sigma_gyro_density, sigma_drift_density, dt_gyro, rng):
    
    # Random-walk variance per step (from drift density)
    q_b_t = (sigma_drift_density**2) * dt_gyro
    
    # White-noise gyro variance per step (from noise density)
    sigma_gyro_t_sq = (sigma_gyro_density**2) / dt_gyro
    
    # Update bias (random walk)
    bias_increment = rng.normal(0, np.sqrt(q_b_t))
    current_bias = previous_bias + bias_increment
    
    # Measurement noise
    measurement_noise = rng.normal(0, np.sqrt(sigma_gyro_t_sq))
    
    # Gyro measurement
    theta_dot_meas = theta_dot_true + current_bias + measurement_noise
    
    # Return both for the next step
    return theta_dot_meas, current_bias
