# simulations/joint_kf_sim.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

import config
from estimation.kf_algo import kf_predict,kf_update
from estimation.kf_matrix import (
    state_transition_cv, measurement_model_encoder, measurement_model_gyro, 
    init_prior, bias_density_to_qb, gyro_density_to_var
)
from sensors.joint_models import simulate_encoder, simulate_gyro

# MODEL FLAG
include_bias = True

# TIME SETTINGS
# Total time
T_total = 6.0
# Encoder time step
dt_enc = 0.01
# Multi-rate (ratio from gyro to encoder)
r = 4
# Gyro time step
dt_gyro = dt_enc / r

# ENCODER SETTINGS
# Dropout probability 
p_dropout = 0.05
# Encoder ticks per revolution
n_ticks = 4096
# Encoder white noise standard deviation (per sample)
sigma_enc = 0.01

# GYRO SETTINGS
# Gyro white-noise standard deviation (continuous-time density)
sigma_gyro_density = 0.02
sigma_gyro_sq = gyro_density_to_var(sigma_gyro_density, dt_gyro)
sigma_gyro = np.sqrt(sigma_gyro_sq) # Convert to per-step standard deviation
# Gyro drift standard deviation (continuous-time density)
sigma_drift_density = 0.002
# Bias per-step variance
q_b = bias_density_to_qb(sigma_drift_density, dt_gyro) 

# PROCESS NOISE
# Angle process noise variance
q_theta = 0.000001
# Angular velocity process noise variance
q_dtheta = 0.0001

# INITIALISATION
x0_theta = 0.0
x0_dtheta = 0.0
x0_bias = 0.0
P0_diag = (1.0, 1.0, 1.0) if include_bias else (1.0, 1.0)

# Random seed
rng = np.random.default_rng(42)

