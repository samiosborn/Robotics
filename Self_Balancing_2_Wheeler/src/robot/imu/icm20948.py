# src/robot/imu/icm20948.py

class ICM20948:
    def __init__(self, i2c_bus, i2c_address, accel_range, gyro_range, axis_map):
        # --- Store Config ---
        # I2C bus
        self._i2c_bus = i2c_bus
        # I2C address
        self._i2c_address = i2c_address
        # Accelerometer range (g)
        self._accel_range = accel_range
        # Gyrometer range (dps)
        self._gyro_range = gyro_range
        # Axis map: dict mapping IMU axes to robot axes
        self._axis_map = axis_map

        # --- Compute Scale Factors ---
        # Accel scale factor
        self._accel_scale = None
        # Gyro scale factor
        self._gyro_scale = None

        # --- Bias ---
        # Accel bias (m/s^2)
        self._bias_accel = [0.0, 0.0, 0.0]
        # Gyro bias (rad/s)
        self._bias_gyro = [0.0, 0.0, 0.0]

        # --- Initialise Hardware ---
        # Open I2C bus
        # Configure accel/gyro ranges
        # Configure LPF or sample rates

    # --- Private Register Access ---
    # Register write
    def _write_register(self, reg, value):
        # Get I2C register
        # Set value
        pass
    # Read register
    def _read_register(self, reg):
        # Get I2C register
        # Read value
        pass

    # --- Public API ---
    # Read IMU
    def read(self):
        # Read Raw Accel and Gyro
        # Apply axis remapping
        # Convert to SI units
        # Subtract bias
        # Return dict: "accel", "gyro", "timestamp"
        pass
    # Calibrate IMU
    def calibrate(self, num_calib_samples):
        # Read IMU in SI
        # Average Error
        # Set Bias for Accel and Gyro
        pass