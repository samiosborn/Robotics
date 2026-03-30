# tests/test_full_io.py
import time
import yaml
import argparse

from src.robot.imu.icm20948 import ICM20948
from src.robot.motors.encoder import Encoder
from src.robot.motors.motor_driver import MotorDriver


# Load central paths config
with open("src/config/paths.yaml", "r") as f:
    PATHS_CFG = yaml.safe_load(f)


# Resolve config paths
IMU_CONFIG_PATH = PATHS_CFG["imu_config"]
MOTOR_CONFIG_PATH = PATHS_CFG["motor_config"]


# Format a float cleanly
def _fmt(value):
    return f"{value: .4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=300)
    parser.add_argument("--pulse-left", action="store_true")
    parser.add_argument("--pulse-right", action="store_true")
    parser.add_argument("--duty", type=float, default=0.12)
    parser.add_argument("--pulse-seconds", type=float, default=0.5)
    args = parser.parse_args()

    imu = None
    encoder = None
    driver = None

    try:
        # Create devices
        imu = ICM20948(IMU_CONFIG_PATH)
        encoder = Encoder(MOTOR_CONFIG_PATH)
        driver = MotorDriver(MOTOR_CONFIG_PATH)

        # Ensure motors are stopped
        driver.stop()

        # Optionally calibrate IMU
        if args.calibrate:
            print("Calibrating IMU")
            imu.calibrate(args.calib_samples)
            print("Calibration complete")

        # Optionally pulse left motor once
        if args.pulse_left:
            print("Pulsing left motor")
            driver.set_duty(abs(args.duty), 0.0)
            time.sleep(args.pulse_seconds)
            driver.stop()
            time.sleep(0.5)

        # Optionally pulse right motor once
        if args.pulse_right:
            print("Pulsing right motor")
            driver.set_duty(0.0, abs(args.duty))
            time.sleep(args.pulse_seconds)
            driver.stop()
            time.sleep(0.5)

        print("Streaming full IO")
        print("Press Ctrl+C to stop")

        while True:
            # Read IMU
            imu_read = imu.read()
            accel = imu_read["accel"]
            gyro = imu_read["gyro"]

            # Read encoders
            count_left, count_right = encoder.get_counts()
            vel_left_rad_s, vel_right_rad_s = encoder.get_angular_velocity()
            vel_left_m_s, vel_right_m_s = encoder.get_linear_velocity()

            # Print one compact line
            print(
                "imu "
                f"pitch_acc={_fmt(accel['pitch'])} "
                f"pitch_gyro={_fmt(gyro['pitch'])}    "
                "enc "
                f"L_count={count_left: 7d} "
                f"L_w={_fmt(vel_left_rad_s)} "
                f"L_v={_fmt(vel_left_m_s)}    "
                f"R_count={count_right: 7d} "
                f"R_w={_fmt(vel_right_rad_s)} "
                f"R_v={_fmt(vel_right_m_s)}"
            )

            # Wait
            time.sleep(args.dt)

    except KeyboardInterrupt:
        print("\nStopped full IO test")

    finally:
        # Stop motors before exit
        if driver is not None:
            driver.stop()

        # Clean up GPIO
        if encoder is not None:
            encoder.cleanup()

        # Close IMU bus
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    main()