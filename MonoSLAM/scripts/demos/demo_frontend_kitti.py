# scripts/demos/demo_frontend_kitti.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_frontend import main as run_demo_frontend
from frontend_common import ROOT


def main() -> None:
    run_demo_frontend(
        default_profile_path=ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml",
        default_output_stem="frontend_kitti",
    )


if __name__ == "__main__":
    main()
