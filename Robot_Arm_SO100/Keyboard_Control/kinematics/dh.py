# dh.py
import numpy as np
import config

# --- 3D TRANSFORMATIONS ---
# Rotation around X-axis
def _rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[1,1] =  ca; T[1,2] = -sa
    T[2,1] =  sa; T[2,2] =  ca
    return T
# Rotation around Z-axis
def _rot_z(t: float) -> np.ndarray:
    ct, st = np.cos(t), np.sin(t)
    T = np.eye(4)
    T[0,0] =  ct; T[0,1] = -st
    T[1,0] =  st; T[1,1] =  ct
    return T
# Translation of X-axis
def _trans_x(a: float) -> np.ndarray:
    T = np.eye(4)
    T[0,3] = a
    return T
# Translation of Z-axis
def _trans_z(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2,3] = d
    return T

# Modified DH (Craig)
def mdh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    # T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    return _rot_z(theta) @ _trans_z(d) @ _trans_x(a) @ _rot_x(alpha)
