# tests/test_left_motor.py
import time
import yaml
import argparse

from src.robot.motors.motor_driver import MotorDriver


# Load central paths config
with open("src/config/paths.yaml", "r") as f:
    PATHS_CFG = yaml.safe_load(f)


# Resolve motor config path
MOTOR_CONFIG_PATH = PATHS_CFG["motor_config"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty", type=float, default=0.15)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    driver = None

    try:
        # Create motor driver
        driver = MotorDriver(MOTOR_CONFIG_PATH)

        # Ensure stopped before test
        driver.stop()

        print("Left motor forward test")
        print("Keep the wheel off the table")

        # Drive left motor forward slowly
        driver.set_duty(abs(args.duty), 0.0)
        time.sleep(args.seconds)

        # Stop
        driver.stop()
        time.sleep(1.0)

        print("Left motor reverse test")

        # Drive left motor reverse slowly
        driver.set_duty(-abs(args.duty), 0.0)
        time.sleep(args.seconds)

        # Stop
        driver.stop()

        print("Left motor test complete")

    except KeyboardInterrupt:
        print("\nStopped left motor test")

    finally:
        # Stop motors before exit
        if driver is not None:
            driver.stop()


if __name__ == "__main__":
    main()