# scripts/diag_pnp_eth3d.py
from __future__ import annotations

from diag_pnp import main as run_diag_pnp
from frontend_eth3d_common import ROOT


def main() -> None:
    run_diag_pnp(
        default_profile_path=ROOT / "configs" / "profiles" / "eth3d_c2.yaml",
        default_output_stem="diag_pnp_eth3d",
    )


if __name__ == "__main__":
    main()
