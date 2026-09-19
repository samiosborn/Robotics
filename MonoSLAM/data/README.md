# Data

Only `two_view/` is tracked. It holds two small images for the two-view
feature-matching demo. ETH3D and KITTI are external datasets: they are
git-ignored, not redistributed, and subject to their own licences. Everything
except the frontend demos and diagnostics runs without them.

Paths are relative to the repository root and match the profiles in
`configs/profiles/`.

## Two-view examples

```text
data/two_view/box.png
data/two_view/box_in_scene.png
```

## ETH3D SLAM (`cables_2_mono`)

Download the monocular `cables_2` sequence from the ETH3D SLAM benchmark
(<https://www.eth3d.net/slam_datasets>) and unpack it so the RGB frames sit at:

```text
data/eth3d/cables_2_mono/cables_2/rgb/<timestamp>.png
```

The loader searches for `rgb/` under the sequence directory and takes each
frame's timestamp from its filename; the accompanying `rgb.txt`,
`groundtruth.txt` and `calibration.txt` are not read. Profile:
`configs/profiles/eth3d_c2.yaml`.

## KITTI odometry (sequence 00)

Download the grayscale odometry set from the KITTI odometry benchmark
(<https://www.cvlibs.net/datasets/kitti/eval_odometry.php>). Only the left
grayscale camera is used:

```text
data/kitti_odometry/sequences/00/image_0/<frame>.png
```

Intrinsics come from `configs/cameras/kitti_odometry.yaml`. Profile:
`configs/profiles/kitti_odometry_00.yaml`.

## Using another location

The scripts read the dataset root from the profile (`dataset.root`). The
frontend demos and `diag_pnp*.py` also accept `--dataset_root`; for
`export_feature_pair.py`, edit a copy of the profile.
