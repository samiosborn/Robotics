# demos/save_waypoint.py
import argparse
import config
from control.motion_controller import MotionController
from storage.waypoints import capture_and_save_waypoint

def main():
    ap = argparse.ArgumentParser(description="Save the current joint angles an a named waypoint.")
    ap.add_argument("--name", required=True, help="Waypoint name")
    ap.add_argument("--no-cache-pose", action="store_true", help="Do not cache pose in the file")
    args = ap.parse_args()

    ctrl = MotionController()
    try:
        w = capture_and_save_waypoint(args.name, ctrl, cache_pose=not args.no_cache_pose)
        print(f"[INFO] Saved waypoint '{args.name}':", w)
    finally:
        ctrl.close()

if __name__ == "__main__":
    main()
