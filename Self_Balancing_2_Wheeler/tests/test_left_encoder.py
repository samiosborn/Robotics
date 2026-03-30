# tests/test_left_encoder.py
import time
import yaml
import argparse

from src.robot.motors.encoder import Encoder


# Load central paths config
with open("src/config/paths.yaml", "r") as f:
    PATHS_CFG = yaml.safe_load(f)


# Resolve motor config path
MOTOR_CONFIG_PATH = PATHS_CFG["motor_config"]


# Format a float cleanly
def _fmt(value):
    return f"{value: .4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    # Create encoder interface
    encoder = Encoder(MOTOR_CONFIG_PATH)

    try:
        print("Streaming left encoder data")
        print("Spin the left wheel by hand first")
        print("Then test it under very slow motor motion")

        while True:
            # Read counts
            count_left, count_right = encoder.get_counts()

            # Read angular velocity
            vel_left_rad_s, vel_right_rad_s = encoder.get_angular_velocity()

            # Read linear velocity
            vel_left_m_s, vel_right_m_s = encoder.get_linear_velocity()

            # Print left side only
            print(
                "left "
                f"count={count_left: 8d} "
                f"omega={_fmt(vel_left_rad_s)} rad/s "
                f"v={_fmt(vel_left_m_s)} m/s"
            )

            # Wait
            time.sleep(args.dt)

    except KeyboardInterrupt:
        print("\nStopped left encoder test")

    finally:
        # Clean up GPIO
        encoder.cleanup()


if __name__ == "__main__":
    main()