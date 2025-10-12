# tests/kin_fk_05_toggle_offset_effect.py
import importlib
import numpy as np
import config
from kinematics.dh import fk_so100

def ee_pos_base(q):
    T = fk_so100(q)
    return T[:3, 3].copy()

def main():
    q = [0, 0, 0, 0, 0]  # straight reach
    x0, z0 = float(config.SHOULDER_OFFSET_X_M), float(config.SHOULDER_OFFSET_Z_M)
    print(f"[INFO] original offsets: x={x0} m, z={z0} m")

    # with offsets (base frame)
    p_with = ee_pos_base(q)
    r_with = float(np.hypot(p_with[0], p_with[1]))
    z_with = float(p_with[2])
    print(f"[INFO] with offsets:  r={r_with:.6f} z={z_with:.6f}")

    # zero the offsets, reload kinematics, re-evaluate
    old_x, old_z = config.SHOULDER_OFFSET_X_M, config.SHOULDER_OFFSET_Z_M
    config.SHOULDER_OFFSET_X_M = 0.0
    config.SHOULDER_OFFSET_Z_M = 0.0
    import kinematics.dh as dh
    importlib.reload(dh)

    T0E_wo = dh.fk_so100(q)
    p_wo = T0E_wo[:3, 3]
    r_wo = float(np.hypot(p_wo[0], p_wo[1]))
    z_wo = float(p_wo[2])
    print(f"[INFO] zero offsets:  r={r_wo:.6f} z={z_wo:.6f}")

    dr = r_with - r_wo
    dz = z_with - z_wo
    print(f"[INFO] deltas:        dr={dr:.6f} (≈{x0}), dz={dz:.6f} (≈{z0})")

    assert abs(dr - x0) < 1e-6, "radius didn’t change by SHOULDER_OFFSET_X_M"
    assert abs(dz - z0) < 1e-6, "z didn’t change by SHOULDER_OFFSET_Z_M"

    # restore constants and kinematics
    config.SHOULDER_OFFSET_X_M = old_x
    config.SHOULDER_OFFSET_Z_M = old_z
    importlib.reload(dh)

    print("[OK] offset toggle behaves as expected (base-frame measurement).")

if __name__ == "__main__":
    main()
