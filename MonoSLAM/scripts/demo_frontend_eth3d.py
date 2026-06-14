# scripts/demo_frontend_eth3d.py
from __future__ import annotations

from demo_frontend import main as run_demo_frontend
from frontend_eth3d_common import ROOT


def main() -> None:
    run_demo_frontend(
        default_profile_path=ROOT / "configs" / "profiles" / "eth3d_c2.yaml",
        default_output_stem="frontend_eth3d",
    )


if __name__ == "__main__":
    main()
