# src/estimation/kf_matrix.py
import numpy as np

# Convert bias random-walk continuous-time density to per step 
def bias_density_to_qb(sigma_drift_density, dt_g):
    return (sigma_drift_density ** 2) * dt_g

# Convert gyro white-noise continuous-time density to per step
def gyro_density_to_var(sigma_gyro_density, dt_g):
    return (sigma_gyro_density ** 2) / dt_g

# State-transition matrix builder (constant velocity, per step)
def state_transition_cv(dt, q_theta, q_dtheta, *, include_bias=False, q_b=None):
    # Input checks
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if q_theta < 0 or q_dtheta < 0:
        raise ValueError("q_theta and q_dtheta must be >= 0")
    if include_bias and (q_b is None or q_b < 0):
        raise ValueError("q_b must be provided and >= 0 when include_bias=True")
    
    # Without bias, x = [theta, dtheta]
    if not include_bias:
        A = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        Q = np.diag([q_theta, q_dtheta])
    else:
        #  With bias, x = [theta, dtheta, b]
        A = np.array([[1.0, dt, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)
        Q = np.diag([q_theta, q_dtheta, q_b])

    return A, Q

# Encoder matrix builder
def measurement_model_encoder(sigma_enc, *, include_bias=False):
    # Variance
    R = np.array([[sigma_enc**2]], dtype=float)
    
    if not include_bias:
        H = np.array([[1.0, 0.0]], dtype=float)
    else:
        # Add extra empty column if bias is included 
        H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        
    return H, R

# Gyro matrix builder
def measurement_model_gyro(sigma_gyro, *, include_bias=False):
    # Variance
    R = np.array([[sigma_gyro**2]], dtype=float)
    
    if not include_bias:
        H = np.array([[0.0, 1.0]], dtype=float)
    else:
        # Include bias term 
        H = np.array([[0.0, 1.0, 1.0]], dtype=float)
        
    return H, R

# Initial state and variance
def init_prior(theta0, dtheta0, *, P_diag, include_bias=False, b0=0.0):
    # Variance
    P0 = np.diag(P_diag)
    
    if not include_bias:
        x0 = np.array([theta0, dtheta0], dtype=float)
        if len(P_diag) != 2:
            raise ValueError("P_diag must be length 2 without bias")
    else:
        # Include bias term
        x0 = np.array([theta0, dtheta0, b0], dtype=float)
        if len(P_diag) != 3:
            raise ValueError("P_diag must be length 3 with bias")
    
    return x0, P0
