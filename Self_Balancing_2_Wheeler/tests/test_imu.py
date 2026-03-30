# tests/test_imu.py
import time
import yaml
import argparse

from src.robot.imu.icm20948 import ICM20948


# Load central paths config
with open("src/config/paths.yaml", "r") as f:
    PATHS_CFG = yaml.safe_load(f)


# Resolve IMU config path
IMU_CONFIG_PATH = PATHS_CFG["imu_config"]


# Format a float cleanly
def _fmt(value):
    return f"{value: .4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=300)
    args = parser.parse_args()

    # Create IMU
    imu = ICM20948(IMU_CONFIG_PATH)

    try:
        # Optionally calibrate first
        if args.calibrate:
            print("Calibrating IMU")
            imu.calibrate(args.calib_samples)
            print("Calibration complete")

        print("Streaming IMU data")
        print("Press Ctrl+C to stop")

        while True:
            # Read IMU
            reading = imu.read()

            # Split accel and gyro
            accel = reading["accel"]
            gyro = reading["gyro"]

            # Print mapped robot axes
            print(
                "accel "
                f"roll={_fmt(accel['roll'])} "
                f"pitch={_fmt(accel['pitch'])} "
                f"yaw={_fmt(accel['yaw'])}    "
                "gyro "
                f"roll={_fmt(gyro['roll'])} "
                f"pitch={_fmt(gyro['pitch'])} "
                f"yaw={_fmt(gyro['yaw'])}"
            )

            # Wait
            time.sleep(args.dt)

    except KeyboardInterrupt:
        print("\nStopped IMU test")

    finally:
        # Close IMU bus
        imu.close()


if __name__ == "__main__":
    main()