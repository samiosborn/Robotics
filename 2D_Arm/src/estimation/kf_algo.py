# src/estimation/kf_algo.py
import numpy as np

# Kalman Filter predict step
def kf_predict(x, P, A, Q):
    # Prior mean: x_{t+1|t} = A x_{t|t}
    x_pred = A @ x
    # Prior covariance: P_{t+1|t} = A P A^T + Q
    P_pred = A @ P @ A.T + Q
    
    return x_pred, P_pred

# Kalman Filter update step
def kf_update(x_pred, P_pred, y, H, R, *, return_gain=False):
    # Handle missing measurement (dropout)
    if y is None or (np.ndim(y) == 0 and not np.isfinite(y)) \
       or (np.ndim(y) > 0 and not np.all(np.isfinite(y))):
        # No update
        if return_gain:
            # Return prior estimate
            return x_pred, P_pred, None, None, None
        return x_pred, P_pred, None, None

    # Innovation (estimate residual): r_t = y - H x_{t|t-1}
    residual = y - (H @ x_pred)

    # Precompute PH^T
    PHt = P_pred @ H.T

    # Innovation covariance: S_t = H P H^T + R
    S = H @ PHt + R

    # Kalman Gain - Solve S * X = PHt^T  ->  X = S^{-1} PHt^T, then transpose
    K = np.linalg.solve(S, PHt.T).T

    # Posterior mean: x_{t|t} = x_{t|t-1} + K r_t
    x_post = x_pred + K @ residual

    # Posterior covariance (Joseph form)
    I = np.eye(P_pred.shape[0])
    IKH = I - K @ H
    # P = (I - K H) P (I - K H)^T + K R K^T
    P_post = IKH @ P_pred @ IKH.T + K @ R @ K.T

    # Enforce symmetry
    P_post = 0.5 * (P_post + P_post.T)

    if return_gain:
        return x_post, P_post, residual, S, K
    return x_post, P_post, residual, S
