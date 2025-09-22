# storage/calibrate_ends.py
import os
import json
from datetime import datetime
from typing import Tuple
import config
from control.feetech_bus import FeetechBus

# Return absolute path to the project's configs/ folder
def _project_configs_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(config.__file__))
    return os.path.join(base_dir, config.CALIBRATION_FOLDER)

# Read raw 2-byte Present_Position for a joint (address 56)
def _read_raw_present(bus: FeetechBus, joint: str) -> int:
    motor_id = config.SERVOS[joint][0]
    raw, result, error = bus.packet_handler.read2ByteTxRx(bus.port_handler, motor_id, 56)
    # scs.COMM_SUCCESS == 0
    if result != 0:
        raise IOError(f"Failed raw read on '{joint}' (id={motor_id}), result={result}, error={error}")
    return int(raw)

# User-friendly labels for min/max guidance per joint  
def _friendly_labels() -> dict:
    return {
        "shoulder_pan": ("rotate fully LEFT", "rotate fully RIGHT"),
        "shoulder_lift": ("move fully DOWN", "move fully UP"),
        "elbow_flex": ("fully BEND", "fully STRAIGHTEN"),
        "wrist_flex": ("tilt fully UP", "tilt fully DOWN"),
        "wrist_roll": ("rotate fully CCW", "rotate fully CW"),
        "gripper": ("fully OPEN", "fully CLOSE"),
    }

# Record end position
def _record_end(bus: FeetechBus, joint: str, label: str) -> Tuple[float, int]:
    # Block until user presses ENTER, then capture (deg, raw)
    while True:
        if config.DEBUG:
            print(f"[ACTION] With torque OFF, {label} the joint '{joint}'.")
        user_input = input("Press ENTER to record, 'r' to redo, 'x' to exit: ").strip().lower()
        # Exit
        if user_input == "x":
            raise KeyboardInterrupt
        # Redo
        if user_input == "r":
            continue
        try:
            # Uses current calibration
            deg = float(bus.read_position(joint))
            # Raw encoder
            raw = _read_raw_present(bus, joint)
            if config.DEBUG:
                print(f"[DEBUG] {joint}: captured {deg:.2f}° (raw {raw})")
            return deg, raw
        except Exception as ex:
            print(f"[ERROR] Read failed on joint '{joint}': {ex}")

def main():
    print("SO-100 Joint End-Stop Calibration")
    print("[SAFETY] Clear the workspace. Support the arm—links may drop with torque off.")
    print("[SAFETY] Move slowly - do NOT force mechanical end-stops.")

    configs_dir = _project_configs_dir()
    os.makedirs(configs_dir, exist_ok=True)

    # Connect to servos
    bus = FeetechBus(port=config.SERVO_PORT, motors=config.SERVOS)
    bus.connect()

    try:
        # Push a placeholder calibration so read_position() can convert raw to deg
        motor_names = list(config.SERVOS.keys())
        placeholder_calib = {
            "motor_names": motor_names,
            "homing_offset": {str(config.SERVOS[name][0]): 0 for name in motor_names},
            "drive_mode": [0] * len(motor_names),
            "calib_mode": ["DEGREE"] * len(motor_names), 
            "start_pos": [0] * len(motor_names),
            "end_pos": [0] * len(motor_names),
        }
        bus.set_calibration(placeholder_calib)

        # Disable torque for hand-guiding
        bus.disable_torque()
        if config.DEBUG:
            print("[INFO] Torque disabled on all joints, so you can hand-guide now.\n")

        labels = _friendly_labels()

        # Capture lists aligned with motor_names order
        start_deg_list, end_deg_list = [], []
        start_raw_list, end_raw_list = [], []

        # Per-joint loop with redo/back/exit
        for joint in motor_names:
            while True:
                print(f"\n--- {joint.upper()} ---")
                redo_entire_joint = False

                # MIN end
                try:
                    start_deg, start_raw = _record_end(bus, joint, labels[joint][0])
                except KeyboardInterrupt:
                    print("[INFO] Calibration cancelled by user.")
                    return

                # MAX end
                while True:
                    if config.DEBUG:
                        print(f"[ACTION] Now {labels[joint][1]} the joint '{joint}'.")
                    user_input = input("Press ENTER to record, 'r' to redo this step, 'b' to redo joint, 'x' to exit: ").strip().lower()
                    if user_input == "x":
                        print("[INFO] Calibration cancelled by user.")
                        return
                    if user_input == "b":
                        redo_entire_joint = True
                        break
                    if user_input == "r":
                        continue
                    try:
                        end_deg = float(bus.read_position(joint))
                        end_raw = _read_raw_present(bus, joint)
                        if config.DEBUG:
                            print(f"[DEBUG] {joint}: captured {end_deg:.2f}° (raw {end_raw})")
                        break
                    except Exception as ex:
                        print(f"[ERROR] Read failed on joint '{joint}': {ex}")

                if redo_entire_joint:
                    print("[INFO] Redoing entire joint…")
                    continue

                # Confirm both ends
                while True:
                    msg = (f"Confirm '{joint}' ends?\n"
                           f" start = {start_deg:.2f}° (raw {start_raw})\n"
                           f" end = {end_deg:.2f}° (raw {end_raw})\n"
                           f"(y = yes / r = redo joint / x = exit): ")
                    confirm = input(msg).strip().lower()
                    if confirm == "y":
                        start_deg_list.append(float(start_deg))
                        end_deg_list.append(float(end_deg))
                        start_raw_list.append(int(start_raw))
                        end_raw_list.append(int(end_raw))
                        break
                    elif confirm == "r":
                        print(f"[INFO] Redoing '{joint}'…")
                        break
                    elif confirm == "x":
                        print("[INFO] Calibration aborted by user.")
                        return
                    else:
                        print("[WARN] Invalid input. Enter 'y', 'r', or 'x'.")

                # If user chose to redo, loop joint again
                if confirm != "y":
                    continue

                # Move on to next joint
                break

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        limits_path = os.path.join(configs_dir, f"limits_{timestamp}.json")
        calib_path  = os.path.join(configs_dir, f"calib_{timestamp}.json")

        # Limits file
        limits_blob = {
            "motor_names": motor_names,
            "start_deg": start_deg_list,
            "end_deg": end_deg_list,
            "start_pos": start_raw_list,
            "end_pos": end_raw_list,
        }
        with open(limits_path, "w", encoding="utf-8") as f:
            json.dump(limits_blob, f, indent=2)
            f.write("\n")
        if config.DEBUG:
            print(f"[SAVED] Limits (for debug) to {limits_path}")

        # Runtime calibration file
        homing_offset = {
            # Centers the motion so 0 is like mid-stroke
            str(config.SERVOS[name][0]): int((s + e) / 2)
            for name, s, e in zip(motor_names, start_raw_list, end_raw_list)
        }
        calib_blob = {
            "motor_names": motor_names,
            "homing_offset": homing_offset,
            "drive_mode": [0] * len(motor_names), 
            "calib_mode": ["DEGREE"] * len(motor_names),
            "start_pos": start_raw_list,
            "end_pos": end_raw_list,
        }
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calib_blob, f, indent=2)
            f.write("\n")
        if config.DEBUG:
            print(f"[SAVED] Calibration (runtime) to {calib_path}")

        # Final print
        print("\n[INFO] Calibration complete.")
        print("To make these active by default, update config.py to:")
        print(f"cALIBRATION_FILENAME = '{os.path.basename(calib_path)}'")
        print(f"LIMITS_FILENAME = '{os.path.basename(limits_path)}'")

    finally:
        try:
            bus.disconnect()
        except Exception:
            pass
        if config.DEBUG:
            print("[INFO] Disconnected from servo bus.")


if __name__ == "__main__":
    main()
