# kinematics/dh.py
import numpy as np
import config

# --- 3D TRANSFORMATIONS ---
# Rotation around X-axis
def _rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    T = np.eye(4, dtype=float)
    T[1,1] =  ca; T[1,2] = -sa
    T[2,1] =  sa; T[2,2] =  ca
    return T

# Rotation around Y-axis
def _rot_y(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    T = np.eye(4, dtype=float)
    T[0,0] =  ca; T[0,2] =  sa
    T[2,0] = -sa; T[2,2] =  ca
    return T

# Rotation around Z-axis
def _rot_z(t: float) -> np.ndarray:
    ct, st = np.cos(t), np.sin(t)
    T = np.eye(4, dtype=float)
    T[0,0] =  ct; T[0,1] = -st
    T[1,0] =  st; T[1,1] =  ct
    return T

# Translation of X-axis
def _trans_x(a: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0,3] = a
    return T

# Translation of Y-axis
def _trans_y(b: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[1,3] = b
    return T

# Translation of Z-axis
def _trans_z(d: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[2,3] = d
    return T

# --- Modified DH (Craig) primitive ---
def mdh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    # T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    return _rot_z(theta) @ _trans_z(d) @ _trans_x(a) @ _rot_x(alpha)

# --- Fixed insert helper for MDH chains ---
# Used by kinematics.forward to inject constant transforms between joints.
def mdh_fixed(kind: str, *args) -> np.ndarray:
    k = kind.lower()
    if k == "tx":
        return _trans_x(float(args[0]))
    if k == "ty":
        return _trans_y(float(args[0]))
    if k == "tz":
        return _trans_z(float(args[0]))
    if k == "mat":
        M = np.array(args[0], dtype=float)
        if M.shape != (4, 4):
            raise ValueError("mdh_fixed('mat', M): M must be 4x4")
        return M
    raise ValueError(f"Unknown mdh_fixed kind '{kind}' (use 'tx','ty','tz','mat')")

# --- SO100 FK (SE(3) chain with explicit J1->J2 shoulder offset) ---
# Joint order: [q1 pan (z), q2 lift (y), q3 elbow (y), q4 wrist_pitch (y), q5 wrist_roll (z)]
def fk_so100(q) -> np.ndarray:
    q1, q2, q3, q4, q5 = [float(v) for v in q]

    # Base -> J1 (pan)
    T01 = _rot_z(q1)

    # Fixed shoulder offset: J1 origin -> J2 origin (do NOT collapse to a single base height)
    T1S = _trans_x(config.SHOULDER_OFFSET_X_M) @ _trans_z(config.SHOULDER_OFFSET_Z_M)

    # J2..J5 chain: links along +x, lifts/pitch about +y, roll about +z
    T12 = _rot_y(q2)
    T23 = _trans_x(config.L1_UPPER_ARM_M) @ _rot_y(q3)
    T34 = _trans_x(config.L2_FOREARM_M)   @ _rot_y(q4)
    T45 = _trans_x(config.L3_WRIST_M)     @ _rot_z(q5)
    T5E = _trans_x(config.TOOL_LENGTH_M)

    return T01 @ T1S @ T12 @ T23 @ T34 @ T45 @ T5E

# Optional: expose intermediate frames for debug/plots
def fk_frames_so100(q):
    q1, q2, q3, q4, q5 = [float(v) for v in q]
    T0  = np.eye(4, dtype=float)
    T01 = _rot_z(q1)
    T0S = T01 @ (_trans_x(config.SHOULDER_OFFSET_X_M) @ _trans_z(config.SHOULDER_OFFSET_Z_M))
    T02 = T0S @ _rot_y(q2)
    T03 = T02 @ (_trans_x(config.L1_UPPER_ARM_M) @ _rot_y(q3))
    T04 = T03 @ (_trans_x(config.L2_FOREARM_M)   @ _rot_y(q4))
    T05 = T04 @ (_trans_x(config.L3_WRIST_M)     @ _rot_z(q5))
    T0E = T05 @ _trans_x(config.TOOL_LENGTH_M)
    return dict(T0=T0, T01=T01, T0S=T0S, T02=T02, T03=T03, T04=T04, T05=T05, T0E=T0E)

# --- quick sanity when run directly ---
if __name__ == "__main__":
    T = fk_so100([0,0,0,0,0])
    p = T[:3,3]
    print("EE @ zeros (m):", p)  # ~ [0.59, 0.0, 0.06] with the given config
