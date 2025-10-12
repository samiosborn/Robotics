# tests/kin_fk_04_numeric_dq1_jacobian.py
import math
import numpy as np
import config
from kinematics.forward import fk_pose

def main():
    n = len(config.MDH_PARAMS)

    # pick a non-degenerate pose
    q = np.zeros(n, dtype=float)
    q[0] = 0.3                        # pan
    if n > 1: q[1] = -0.4             # lift
    if n > 2: q[2] =  0.2             # elbow
    if n > 3: q[3] = -0.1             # wrist pitch
    if n > 4: q[4] =  0.15            # wrist roll

    # fk at nominal
    p0, _ = fk_pose(q.tolist())

    # finite difference wrt q1
    h = 1e-6
    qp = q.copy(); qn = q.copy()
    qp[0] += h; qn[0] -= h
    pp, _ = fk_pose(qp.tolist())
    pn, _ = fk_pose(qn.tolist())
    dp_num = (pp - pn) / (2*h)

    # analytic: z-hat × p
    zhat = np.array([0.0, 0.0, 1.0])
    dp_ana = np.cross(zhat, p0)

    err = np.linalg.norm(dp_num - dp_ana)
    print(f"[INFO] ||dp_num - dp_ana|| = {err:.3e}")
    print(f"[INFO] dp_num = {dp_num}")
    print(f"[INFO] dp_ana = {dp_ana}")

    assert err < 1e-6, "∂p/∂q1 numerical derivative doesn’t match z×p"
    print("[OK] dq1 Jacobian check passes (z-hat cross p).")

if __name__ == "__main__":
    main()
