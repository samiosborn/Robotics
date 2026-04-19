# scripts/two_view_synthetic.py
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from synthetic.two_view import generate_two_view_scene
from geometry.essential import essential_from_pose, decompose_essential
from geometry.triangulation import disambiguate_pose_cheirality
from geometry.rotation import angle_between_rotmats
from geometry.pose import angle_between_translations


def main():
    parser = argparse.ArgumentParser()
    parser.parse_known_args()

    # Generate scene data
    scene_data = generate_two_view_scene(n_points=1)

    # Unpack
    R_true = scene_data["R"]
    t_true = scene_data["t"]
    x1 = scene_data["x1"]
    x2 = scene_data["x2"]
    K1 = scene_data["K1"]
    K2 = scene_data["K2"]

    # Essential matrix
    E = essential_from_pose(R_true, t_true)

    # Decompose essential matrix
    candidate_poses = decompose_essential(E)

    # Select valid pose
    R_est, t_est, _ = disambiguate_pose_cheirality(candidate_poses, K1, K2, x1, x2)

    # Error in rotation
    R_error = angle_between_rotmats(R_est, R_true)
    print("Rotation error (radians): ", np.round(R_error, 3))

    # Error in translation
    t_error = angle_between_translations(t_est, t_true)
    print("Translation error (radians): ", np.round(t_error, 3))


if __name__ == "__main__":
    main()
