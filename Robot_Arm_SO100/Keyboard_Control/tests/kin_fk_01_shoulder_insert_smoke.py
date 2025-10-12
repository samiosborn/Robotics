# tests/kin_fk_01_shoulder_insert_smoke.py
import numpy as np
import config
from kinematics.forward import fk_all_frames

def _T_shoulder_from_config():
    T = np.eye(4, dtype=float)
    T[0, 3] = float(config.SHOULDER_OFFSET_X_M)
    T[2, 3] = float(config.SHOULDER_OFFSET_Z_M)
    return T

def main():
    n = len(config.MDH_PARAMS)
    q = [0.0] * n

    Ts = fk_all_frames(q)

    # frames: T0 + each joint + extra after J1 + tool = n + 3
    expected_len = n + 3
    assert len(Ts) == expected_len, f"frames mismatch: got {len(Ts)}, expected {expected_len}"

    T01 = Ts[1]            # after J1 rotation
    T_after_offset = Ts[2] # we expect this to be T01 @ T_shoulder
    T_expected = T01 @ _T_shoulder_from_config()

    if not np.allclose(T_after_offset, T_expected, atol=1e-9, rtol=0):
        raise AssertionError("J1→J2 fixed shoulder offset not found at expected spot.")

    print("[OK] shoulder offset is inserted after J1, frame count looks good.")

if __name__ == "__main__":
    main()

