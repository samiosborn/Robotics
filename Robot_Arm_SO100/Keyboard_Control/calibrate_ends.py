# calibrate_ends.py
import os
import json
import config
from feetech_bus import FeetechBus
from datetime import datetime

def main():
    bus = FeetechBus(config.SERVO_PORT, config.SERVOS)
    bus.connect()
    bus.disable_torque()

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

    start_deg_list = []
    end_deg_list = []
    start_raw_list = []
    end_raw_list = []

    print("[INFO] Starting joint limit calibration. Use your hands to move joints gently.\n")

    direction_labels = {
        "shoulder_pan": ("rotate fully left", "rotate fully right"),
        "shoulder_lift": ("move fully down", "move fully up"),
        "elbow_flex": ("fully bend", "fully straighten"),
        "wrist_flex": ("tilt fully up", "tilt fully down"),
        "wrist_roll": ("rotate fully counter-clockwise", "rotate fully clockwise"),
        "gripper": ("fully open", "fully close"),
    }

    for joint in motor_names:
        # Full joint loop
        while True:
            print(f"\n--- {joint.upper()} ---")
            label_min, label_max = direction_labels[joint]
            redo_joint = False

            # Start bound
            while True:
                print(f"[ACTION] {label_min}.")
                user_input = input("Press ENTER to record, 'r' to redo, 'x' to exit: ").strip().lower()
                if user_input == "x":
                    print("[INFO] Calibration aborted by user.")
                    bus.disconnect()
                    return
                elif user_input == "r":
                    continue
                else:
                    try:
                        start_deg = bus.read_position(joint)
                        start_raw = bus.packet_handler.read2ByteTxRx(bus.port_handler, config.SERVOS[joint][0], 56)[0]
                        if config.DEBUG:
                            print(f"[DEBUG] {joint} start recorded: {start_deg:.1f}°")
                        break
                    except Exception as e:
                        print(f"[ERROR] Failed to read joint '{joint}': {e}")

            # End bound
            while True:
                print(f"[ACTION] {label_max}.")
                user_input = input("Press ENTER to record, 'r' to redo, 'b' to go back, 'x' to exit: ").strip().lower()
                if user_input == "x":
                    print("[INFO] Calibration aborted by user.")
                    bus.disconnect()
                    return
                elif user_input == "b":
                    print("[INFO] Going back to redo entire joint.")
                    redo_joint = True
                    break
                elif user_input == "r":
                    continue
                else:
                    try:
                        end_deg = bus.read_position(joint)
                        end_raw = bus.packet_handler.read2ByteTxRx(bus.port_handler, config.SERVOS[joint][0], 56)[0]
                        if config.DEBUG:
                            print(f"[DEBUG] {joint} end recorded: {end_deg:.1f}°")
                        break
                    except Exception as e:
                        print(f"[ERROR] Failed to read joint '{joint}': {e}")

            if redo_joint:
                # Redo this joint from the top
                continue

            # Confirm both readings
            while True:
                confirm = input(f"Confirm calibration for '{joint}'? (y = yes / r = redo / x = exit): ").strip().lower()
                if confirm == "y":
                    start_deg_list.append(start_deg)
                    start_raw_list.append(int(start_raw))
                    end_deg_list.append(end_deg)
                    end_raw_list.append(int(end_raw))
                    # Exit confirmation loop and move to next joint
                    break
                elif confirm == "r":
                    print(f"[INFO] Redoing '{joint}'...")
                    redo_joint = True
                    break
                elif confirm == "x":
                    print("[INFO] Calibration aborted by user.")
                    bus.disconnect()
                    return
                else:
                    print("[WARN] Invalid input. Enter 'y', 'r', or 'x'.")

            if redo_joint:
                # Re-run this joint entirely
                continue
            # Move to next joint
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("configs", exist_ok=True)

    # Save original raw limits file
    limits_path = os.path.join("configs", f"limits_{timestamp}.json")
    raw_data = {
        "motor_names": motor_names,
        "start_deg": start_deg_list,
        "end_deg": end_deg_list,
        "start_pos": start_raw_list,
        "end_pos": end_raw_list,
    }
    with open(limits_path, "w") as f:
        json.dump(raw_data, f, indent=4)
    print(f"[SAVED] Raw limits file written to: {limits_path}")

    # Save runtime calibration
    homing_offset = {
        str(config.SERVOS[m][0]): int((s + e) / 2)
        for m, s, e in zip(motor_names, start_raw_list, end_raw_list)
    }

    calib_data = {
        "motor_names": motor_names,
        "homing_offset": homing_offset,
        "drive_mode": [0] * len(motor_names),
        "calib_mode": ["DEGREE"] * len(motor_names),
        "start_pos": start_raw_list,
        "end_pos": end_raw_list,
    }
    calib_path = os.path.join("configs", f"calib_{timestamp}.json")
    with open(calib_path, "w") as f:
        json.dump(calib_data, f, indent=4)
    print(f"[SAVED] Runtime calib file written to: {calib_path}")

    bus.disconnect()

if __name__ == "__main__":
    main()
