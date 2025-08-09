# joint_models.py
import numpy as np

# Number of ticks per revolution
n_ticks = 4096

# Probability of dropout
p_dropout = 0.05

# Observation noise standard deviation
sigma_r = 1.0

# Random Number Generator
rng = np.random.default_rng(seed=42)

# Simulate Noisy Encoder
def simulate_encoder(theta_true, p_dropout, n_ticks, sigma_r, rng):
    
    # Sample from Uniform distribution in [0, 1)
    bernoulli_sample = rng.random()

    # Dropout probability
    if bernoulli_sample < p_dropout:
        # Return NaN as dropout
        return np.nan
    else: 
        # Sample Gaussian noise
        measurement_noise = rng.normal(0, sigma_r)
        # Actual measurement
        theta_measurement = theta_true + measurement_noise
        # Theta tick delta
        theta_delta = 2* np.pi / n_ticks 
        # Clipped measurement
        theta_measurement = np.round(theta_measurement / theta_delta) * theta_delta
        # Return measurement
        return theta_measurement

# Simulate Noisy Gyrometer [TBC]

