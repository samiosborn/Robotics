# adapters/keyboard_control.py
import keyboard
import config
from control.motion_controller import run_step_loop

# Step multipliers
def step_multiplier() -> float:
    if any(keyboard.is_pressed(k) for k in config.KEYBOARD_FINE_MODS):
        return config.KEYBOARD_FINE_MULTIPLIER
    if any(keyboard.is_pressed(k) for k in config.KEYBOARD_FAST_MODS):
        return config.KEYBOARD_FAST_MULTIPLIER
    return 1.0

# Get command (joint_name, step_deg) for a single step
def get_active_command():
    # Fast-path quit keys
    if keyboard.is_pressed("esc") or keyboard.is_pressed("z"):
        return ("__QUIT__", 0.0)

    # Scan keys
    for key, binding in config.KEYBOARD_BINDINGS.items():
        if not keyboard.is_pressed(key):
            continue
        if binding == "quit":
            return ("__QUIT__", 0.0)
        joint, direction = binding
        step = direction * config.STEP_SIZE * step_multiplier()
        return (joint, step)

    return None

# Keyboard teleop loop until exit
def main_loop(step_sleep: float = 0.05):
    print("Keyboard teleop: hold SHIFT=coarse; CTRL/ALT=fine; 'z' or ESC to quit.")
    try:
        run_step_loop(get_active_command, step_sleep=step_sleep)
    except PermissionError:
        if config.DEBUG:
            # Common on Windows (if not elevated to admin)
            print("[ERROR] Permission denied capturing keyboard events.")
            print("On Windows, open your terminal as Administrator.")
        raise
    except KeyboardInterrupt:
        # Exit
        pass


if __name__ == "__main__":
    main_loop()
