# tests/test_motor_driver.py
import time
import yaml

from src.robot.motors.motor_driver import MotorDriver


# Load central paths config
with open("src/config/paths.yaml", "r") as f:
    PATHS_CFG = yaml.safe_load(f)


# Resolve motor config path
MOTOR_CONFIG_PATH = PATHS_CFG["motor_config"]


def main():
    driver = None

    try:
        # Create motor driver
        driver = MotorDriver(MOTOR_CONFIG_PATH)

        # Send a clean stop command
        driver.stop()

        print("Motor driver initialised successfully")
        print("Stop command sent successfully")

        # Pause briefly
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped motor driver test")

    finally:
        # Stop motors before exit
        if driver is not None:
            driver.stop()


if __name__ == "__main__":
    main()