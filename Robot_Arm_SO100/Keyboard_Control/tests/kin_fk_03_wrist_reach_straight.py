# tests/kin_fk_03_wrist_reach_straight.py
import numpy as np
import config
from kinematics.forward import fk_all_frames

def main():
    n = len(config.MDH_PARAMS)
    q = [0.0] * n  # q1..q5 = 0

    Ts = fk_all_frames(q)

    T_shoulder = Ts[2]    # world→shoulder
    T_wrist    = Ts[-2]   # world→wrist-center (tool not applied yet)

    # shoulder→wrist
    T_s_to_w = np.linalg.inv(T_shoulder) @ T_wrist
    p_local  = T_s_to_w[:3, 3]

    L1 = float(config.L1_UPPER_ARM_M)
    L2 = float(config.L2_FOREARM_M)
    L3 = float(config.L3_WRIST_M)
    expected = np.array([L1 + L2 + L3, 0.0, 0.0], dtype=float)

    err = np.linalg.norm(p_local - expected)
    print(f"[INFO] wrist (in shoulder frame) = {p_local}, expected ~ {expected}")
    print(f"[INFO] position error norm = {err:.6e} m")

    # be a little relaxed numerically (MDH alpha choices can add tiny numeric noise)
    assert err < 1e-9, "straight reach length from shoulder doesn’t match L1+L2+L3"
    print("[OK] straight reach to wrist matches L1+L2+L3 along shoulder x-axis.")

if __name__ == "__main__":
    main()
