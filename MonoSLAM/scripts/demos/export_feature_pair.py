from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend_common import ROOT, load_pil_greyscale, load_runtime_cfg
from datasets.loader import load_sequence
from features.pipeline import detect_and_describe_image
from features.viz import draw_matches
from slam.matching import match_frames, matched_keypoints_xy
from slam.two_view_consensus import estimate_fundamental_consensus


# Use Pillow's bundled font to avoid system-font dependencies.
def label_image(image, title, detail, labels):
    canvas = Image.new("RGB", (image.width, image.height + 100), (20, 25, 30))
    canvas.paste(image, (0, 100))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), title, font=ImageFont.load_default(size=23), fill="white")
    draw.text((18, 43), detail, font=ImageFont.load_default(size=16), fill=(195, 205, 215))
    for x, label in zip((18, image.width // 2 + 18), labels):
        draw.text((x, 76), label, font=ImageFont.load_default(size=15), fill=(195, 205, 215))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Export a dataset feature pair using the current frontend algorithms.")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/profiles/eth3d_c2.yaml")
    parser.add_argument("--i0", type=int, required=True)
    parser.add_argument("--i1", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/website_exports")
    parser.add_argument("--max-draw", type=int, default=48)
    parser.add_argument("--draw-seed", type=int, default=0)
    args = parser.parse_args()
    if args.i0 < 0 or args.i1 <= args.i0 or args.max_draw < 1 or args.draw_seed < 0:
        parser.error("Require 0 <= i0 < i1, max-draw >= 1 and draw-seed >= 0")

    cfg, _ = load_runtime_cfg(args.profile.resolve())
    dataset = cfg["dataset"]
    seq = load_sequence(dataset["name"], ROOT / dataset["root"], dataset["seq"])
    indices = (args.i0, args.i1)
    records = [seq.frame_info(i) for i in indices]
    images = [load_pil_greyscale(r.path) for r in records]
    features = [detect_and_describe_image(seq.get(i)[0], cfg["features"]) for i in indices]
    matches = match_frames(*features)
    xy0, xy1 = matched_keypoints_xy(*features, matches)
    F, mask, stats = estimate_fundamental_consensus(xy0, xy1, cfg)
    if F is None or mask is None or not stats.get("refit"):
        raise RuntimeError(f"Geometric verification failed: {stats}")

    # Select independently of geometric status; retain the same subset for comparison.
    n_raw = len(matches.ia)
    selected = np.sort(np.random.default_rng(args.draw_seed).choice(
        n_raw, size=min(args.max_draw, n_raw), replace=False,
    ))
    selected_inliers = selected[mask[selected]]
    n_inliers = int(mask.sum())
    counts = [len(f.kps_xy) for f in features]
    labels = [f"Frame {i}  |  t = {r.timestamp:.6f} s" for i, r in zip(indices, records)]
    name = dataset["name"]
    sequence_label = f"{name.upper()} / {dataset['seq']}"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Show every descriptor-bearing multiscale keypoint in base-image coordinates.
    panels = []
    for image, feats in zip(images, features):
        panel = image.convert("RGB")
        draw = ImageDraw.Draw(panel)
        for x, y in feats.kps_xy:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(40, 220, 235), width=1)
        panels.append(panel)
    pair = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
    pair.paste(panels[0], (0, 0))
    pair.paste(panels[1], (panels[0].width, 0))
    keypoint_path = out / f"{name}_keypoints_pair.png"
    label_image(pair, "Detected keypoints", f"{sequence_label}  |  {counts[0]} / {counts[1]} multiscale features; all shown", labels).save(keypoint_path)

    for suffix, chosen, title, detail in (
        ("raw", selected, "Tentative descriptor matches",
         f"{n_raw} total; fixed random sample of {len(selected)} shown"),
        ("inliers", selected_inliers, "Geometrically verified matches",
         f"{n_inliers} / {n_raw} fundamental-matrix inliers; {len(selected_inliers)} survivors of the same sample shown"),
    ):
        path = out / f"{name}_matches_{suffix}.png"
        draw_matches(*images, features[0].kps_xy, features[1].kps_xy,
                     matches.ia[chosen], matches.ib[chosen], path,
                     max_draw=len(chosen), r=2, colour=(40, 220, 235))
        with Image.open(path) as rendered:
            labelled = label_image(rendered, title, detail, labels)
        labelled.save(path)

    # Record resolved inputs, configuration and display indices without large arrays.
    matching_defaults = {
        key: value.default for key, value in inspect.signature(match_frames).parameters.items()
        if value.default is not inspect.Parameter.empty
    }
    metadata = {
        "dataset": seq.name,
        "frames": [
            {"index": i, "frame_id": r.frame_id, "timestamp": r.timestamp,
             "image_path": str(r.path.relative_to(ROOT)),
             "sha256": hashlib.sha256(r.path.read_bytes()).hexdigest()}
            for i, r in zip(indices, records)
        ],
        "profile": str(args.profile.resolve().relative_to(ROOT)),
        "feature_configuration": cfg["features"],
        "feature_defaults_source": "features.pipeline.detect_and_describe_image",
        "matching_configuration": matching_defaults,
        "keypoint_counts": counts,
        "keypoint_definition": "All descriptor-bearing multiscale detections, mapped to base-image coordinates; no cross-level deduplication",
        "raw_matches": n_raw,
        "verified_inliers": n_inliers,
        "verification": {
            "method": "Normalised 8-point fundamental-matrix RANSAC, Sampson distance, guarded inlier refit",
            "configuration": cfg["ransac"]["F"],
            "seed": cfg["ransac"].get("seed", 0),
            "statistics": stats,
            "fundamental_matrix": F.tolist(),
        },
        "visualisation": {
            "greyscale": True,
            "selection": "Uniform random sample without replacement before geometric filtering; verified panel shows its surviving inliers",
            "seed": args.draw_seed,
            "max_draw": args.max_draw,
            "raw_match_indices": selected.tolist(),
            "inlier_match_indices": selected_inliers.tolist(),
        },
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "scripts/demos/export_feature_pair.py", "src/features/pipeline.py",
                "src/features/multiscale.py", "src/features/matching.py",
                "src/features/viz.py", "src/slam/matching.py",
                "src/slam/two_view_consensus.py", "src/geometry/fundamental.py", "uv.lock",
            )
        },
        "command": "uv run python " + shlex.join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "versions": {"numpy": np.__version__, "pillow": Image.__version__},
    }
    (out / f"{name}_feature_matching_results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Keypoints: {counts}; raw matches: {n_raw}; verified inliers: {n_inliers}")
    print(f"Displayed: {len(selected)} raw, {len(selected_inliers)} inliers; outputs: {out}")


if __name__ == "__main__":
    main()
