# storage/waypoints.py
import os
import json
import time
from typing import Dict, List
import numpy as np
import config
from kinematics.forward import fk_pose

# Default storage path for saved waypoints
WAYPOINTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config.CALIBRATION_FOLDER))
WAYPOINTS_PATH = os.path.join(WAYPOINTS_DIR, config.WAYPOINTS_FILENAME)

# Ensure the default waypoints folder & file exist
def _ensure_store():
    os.makedirs(WAYPOINTS_DIR, exist_ok=True)
    if not os.path.exists(WAYPOINTS_PATH):
        with open(WAYPOINTS_PATH, "w", encoding="utf-8") as f:
            # Create a new empty JSON if file doesn't exist
            json.dump({}, f, indent=2)
            f.write("\n")

# Load all waypoints as a dict
def load_waypoints() -> Dict:
    _ensure_store()
    with open(WAYPOINTS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Return empty JSON
            return {}

# Save waypoints (dict of name -> waypoint dict) to disk
def save_waypoints(blob: Dict) -> None:
    _ensure_store()
    # Save waypoint initially to temp file
    tmp_path = WAYPOINTS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
        f.write("\n")
        # Push data in buffer
        f.flush()
        # Push buffer data to disk
        os.fsync(f.fileno())
    # Atomic metadata update
    os.replace(tmp_path, WAYPOINTS_PATH)

# Save or update a named waypoint in the JSON file
def save_single_waypoint(name: str, waypoint: Dict) -> None:
    blob = load_waypoints()
    # If name doesn't exist, add new entry. Otherwise, update it
    blob[name] = waypoint
    save_waypoints(blob)

# Get pose [x,y,z,r,p,y] from a waypoint dict
def pose_from_waypoint(w: dict, prefer_cache: bool = True) -> np.ndarray:
    if prefer_cache and "pose" in w and w["pose"] is not None:
        return np.array(w["pose"], dtype=float)
    # Recompute pose
    q_deg = w["q_deg"]
    q_vec_deg = [q_deg[j] for j in config.JOINT_ORDER]
    q_rad = np.deg2rad(q_vec_deg)
    p, rpy = fk_pose(q_rad)
    return np.hstack([p, rpy])

# Build a waypoint dict from joint angles
def build_waypoint_from_qdeg(q_deg: Dict[str, float], cache_pose: bool = True) -> Dict:
    # Order by joint
    q_deg_ordered = {j: float(q_deg[j]) for j in config.JOINT_ORDER}
    # Add timestamp: "YYYY-MM-DD HH:MM:SS"
    w = {
        "q_deg": q_deg_ordered,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if cache_pose:
        q_vec_deg = [q_deg_ordered[j] for j in config.JOINT_ORDER]
        q_rad = np.deg2rad(q_vec_deg)
        p, rpy = fk_pose(q_rad)
        # Add pose component
        w["pose"] = [float(p[0]), float(p[1]), float(p[2]),
             float(rpy[0]), float(rpy[1]), float(rpy[2])]
    else:
        w["pose"] = None
    return w

# Capture current joint angles and save
def capture_and_save_waypoint(name: str,
                              controller,
                              cache_pose: bool = True) -> Dict:
    # Read current joint angles in degrees from the hardware
    q_deg = {}
    for j in config.JOINT_ORDER:
        q_deg[j] = float(controller.bus.read_position(j))
    # Build waypoint
    w = build_waypoint_from_qdeg(q_deg, cache_pose=cache_pose)
    save_single_waypoint(name, w)
    return w

# Get waypoint from it's name
def get_waypoint(name: str) -> Dict:
    blob = load_waypoints()
    try:
        return blob[name]
    except KeyError as e:
        raise KeyError(f"Waypoint '{name}' not found in {WAYPOINTS_PATH}") from e

# Get waypoint's pose from it's name
def get_waypoint_pose(name: str, prefer_cache: bool = True) -> np.ndarray:
    # Get waypoint
    w = get_waypoint(name)
    return pose_from_waypoint(w, prefer_cache)

# List all waypoints by name
def list_waypoint_names() -> List[str]:
    return sorted(load_waypoints().keys())

# Delete waypoints by name
def delete_waypoint(name: str) -> bool:
    blob = load_waypoints()
    existed = name in blob
    if existed:
        del blob[name]
        save_waypoints(blob)
    return existed

# Rename waypoint (with overwrite)
def rename_waypoint(old: str, new: str, overwrite: bool = False) -> None:
    blob = load_waypoints()
    if old not in blob:
        raise KeyError(f"Waypoint '{old}' not found.")
    if (new in blob) and not overwrite:
        raise ValueError(f"Waypoint '{new}' already exists. Use overwrite=True to replace.")
    blob[new] = blob.pop(old)
    save_waypoints(blob)

# Clear all saved waypoints
def clear_waypoints() -> None:
    # Makes sure waypoints folder/file exist
    _ensure_store()
    # Overwrite file with empty JSON
    save_waypoints({})
    if config.DEBUG:
        print(f"[INFO] Cleared all waypoints in {WAYPOINTS_PATH}")

# Prune pose cache
def prune_pose_cache(names: List[str] | None = None) -> None:
    # Set pose=None for selected (or all) waypoints to force recompute later
    blob = load_waypoints()
    target_names = names or list(blob.keys())
    for n in target_names:
        if n in blob:
            blob[n]["pose"] = None
    save_waypoints(blob)

# Refresh pose for all waypoints
def refresh_all_poses(names: List[str] | None = None) -> None:
    blob = load_waypoints()
    target_names = names or list(blob.keys())
    for n in target_names:
        w = blob.get(n)
        if not w or "q_deg" not in w or w["q_deg"] is None:
            continue
        # Compute pose
        q_vec_deg = [w["q_deg"][j] for j in config.JOINT_ORDER]
        q_rad = np.deg2rad(q_vec_deg)
        p, rpy = fk_pose(q_rad)
        w["pose"] = [float(p[0]), float(p[1]), float(p[2]),
                     float(rpy[0]), float(rpy[1]), float(rpy[2])]
    save_waypoints(blob)
