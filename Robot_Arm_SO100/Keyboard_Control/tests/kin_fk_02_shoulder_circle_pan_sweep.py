# tests/kin_fk_02_shoulder_circle_pan_sweep.py
import numpy as np
import config
from kinematics.dh import fk_frames_so100

def main():
    rs, zs = [], []
    for q1 in np.linspace(-np.pi, np.pi, 73):  # pan sweep
        fr = fk_frames_so100([q1, 0, 0, 0, 0])
        pS = fr["T0S"][:3, 3]                    # <-- shoulder origin in base frame
        rs.append(float(np.hypot(pS[0], pS[1])))
        zs.append(float(pS[2]))

    r_target = float(config.SHOULDER_OFFSET_X_M)
    z_target = float(config.SHOULDER_OFFSET_Z_M)

    r_err = max(abs(r - r_target) for r in rs)
    z_err = max(abs(z - z_target) for z in zs)

    print(f"[INFO] r_target={r_target:.6f}m, z_target={z_target:.6f}m")
    print(f"[INFO] max radius error = {r_err:.6e} m")
    print(f"[INFO] max z error      = {z_err:.6e} m")

    assert r_err < 1e-9 and z_err < 1e-9, \
        "shoulder frame is not staying on the expected circle/height"
    print("[OK] shoulder origin traces the expected circle.")

if __name__ == "__main__":
    main()
