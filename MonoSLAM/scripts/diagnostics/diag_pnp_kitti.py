# scripts/diagnostics/diag_pnp_kitti.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diag_pnp import main as run_diag_pnp
from frontend_common import ROOT


def main() -> None:
    run_diag_pnp(
        default_profile_path=ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml",
        default_output_stem="diag_pnp_kitti",
    )


if __name__ == "__main__":
    main()
