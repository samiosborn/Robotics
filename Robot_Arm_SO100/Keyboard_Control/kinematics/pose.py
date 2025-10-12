# kinematics/pose.py
import numpy as np

# RPY to rotation matrix (ZYX order: yaw, then pitch, then roll)
def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
    return Rz @ Ry @ Rx

# Rotation matrix to RPY (ZYX order)
def R_to_rpy(R: np.ndarray) -> tuple[float,float,float]:
    sy = -R[2,0]
    cy = np.sqrt(max(0.0, 1.0 - sy*sy))
    # Protect against singularities
    if cy > 1e-9:
        pitch = np.arctan2(sy, cy)
        roll  = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(sy, cy)
        roll  = 0.0
        yaw   = np.arctan2(-R[0,1], R[1,1])
    return (roll, pitch, yaw)

# SO(3) hat - turns a 3-vector into a skew-symmetric matrix
def hat(omega: np.ndarray) -> np.ndarray:
    x, y, z = omega
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=float)

# SO(3) vee - pulls the 3-vector back out of a skew-symmetric matrix
def vee(W: np.ndarray) -> np.ndarray:
    return np.array([W[2,1]-W[1,2], W[0,2]-W[2,0], W[1,0]-W[0,1]]) * 0.5

# SO(3) log map - returns a rotation vector (axis * angle)
def so3_log(R: np.ndarray) -> np.ndarray:
    # cos(theta) = (trace(R) - 1) / 2
    tr = (np.trace(R) - 1.0) * 0.5
    c = np.clip(tr, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    # Recover skew-symmetrix matrix
    W = (R - R.T) / (2.0 * np.sin(th))
    # Convert skew-symmetrix matrix into 3-vector and multiply by theta
    return vee(W) * th

# SO(3) exponential map for a rotation vector w
def so3_exp(w: np.ndarray) -> np.ndarray:
    # Angle from norm
    th = float(np.linalg.norm(w))
    # Skew symmetrix matrix version
    W = hat(w)
    # For tiny angles
    if th < 1e-8:
        # Avoids dividing by zero
        return np.eye(3) + W + 0.5 * (W @ W)
    # Unnormalised Rodrigues rotation formula
    A = np.sin(th) / th
    B = (1.0 - np.cos(th)) / (th * th)
    # First 2 terms of Taylor series
    return np.eye(3) + A * W + B * (W @ W)