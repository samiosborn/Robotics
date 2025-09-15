# keyboard_control.py
import keyboard
import time
import config
from motion_controller import run_step_loop

# Check if any control key is pressed
def get_active_command():
    # Get keyboard inputs
    for key, command in config.KEYBOARD_BINDINGS.items():
        if keyboard.is_pressed(key):
            # Quit sentinel for the shared step-loop runner
            if command == "quit":
                return ("__QUIT__", 0.0)
            # Map key to (joint, signed step degrees)
            joint, direction = command
            return (joint, direction * config.STEP_SIZE)
    # No key active = no command
    return None

# Run the main teleop loop
def main_loop():
    # Info for the user
    print("Teleop running. Press 'z' to quit. Click away from terminal and code.")
    # Shared step loop handles controller lifecycle and pacing
    run_step_loop(get_active_command, step_sleep=0.05)

# Entry
if __name__ == "__main__":
    main_loop()
