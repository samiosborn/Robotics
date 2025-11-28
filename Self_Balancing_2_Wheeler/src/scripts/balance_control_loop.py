# src/scripts/balance_control_loop.py
import yaml
import time

# Load Config Paths
with open("src/config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

# Configs
imu_path = paths["imu_config"]
control_path = paths["control_config"]
motor_path = paths["motor_config"]

# Import Modules
from robot.imu.icm20948 import ICM20948
from robot.imu.complementary_filter import ComplementaryFilter
from robot.control.pid import PID
from robot.control.balance_controller import BalanceController
from robot.motors.motor_driver import MotorDriver
from robot.motors.encoder import Encoder

# Instantiate Modules
imu = ICM20948(imu_path)
filter = ComplementaryFilter(control_path)
pid = PID(control_path)
controller = BalanceController(control_path, pid)
motors = MotorDriver(motor_path)
encoders = Encoder(motor_path)

# Safe startup
def safe_startup(imu, filter, motors, encoders):
    # Stop motors
    motors.stop()
    # Give IMU time to stabilise
    time.sleep(0.2)
    # IMU Read
    reading = imu.read()
    accel_reading = reading["accel"]
    gyro_reading = reading["gyro"]
    # Initialise complementary filter
    filter.update(accel_reading, gyro_reading, 0.0)
    # Initialise encoders
    encoders.reset_counts()
    encoders.reset_velocities()
    # Previous time
    prev_time = time.time()
    return prev_time

# Main Loop
def main():
    # Safe Startup
    prev_time = safe_startup(imu, filter, motors, encoders)

    # Loop
    while True:
        # IMU Read
        reading = imu.read()
        accel_reading = reading["accel"]
        gyro_reading = reading["gyro"]
        now = reading["timestamp"]
        # Update timestep    
        dt = now - prev_time
        prev_time = now
        # Update complementary filter
        theta = filter.update(accel_reading, gyro_reading, dt)
        theta_dot = filter.theta_dot
        # Safety cutoff 
        if abs(theta) > 0.6:
            motors.stop()
            continue
        # Update controller
        left_duty, right_duty = controller.update(theta, theta_dot, dt)
        # Set duty
        motors.set_duty(left_duty, right_duty)
        # Limit loop frequency
        time.sleep(0.001)


if __name__ == "__main__":
    main()
