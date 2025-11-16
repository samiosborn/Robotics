# src/robot/imu/icm20948.py
import smbus2
import yaml

class ICM20948:
    def __init__(self, yaml_config_path):
        # --- Load YAML Config ---
        with open(yaml_config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # --- Class Variables ---
        # I2C bus number
        self._i2c_bus = cfg["i2c_bus"]
        # I2C address
        self._i2c_address = cfg["i2c_address"]
        # Bank selector register
        self._reg_bank_sel = cfg["reg_bank_sel"]
        # Gyro configs
        self._gyro_cfg = cfg["gyro_configs"]
        # Accel configs
        self._accel_cfg = cfg["accel_configs"]
        # Axis map: dict mapping IMU axes to robot axes
        self._axis_map = cfg["axis_map"]

        # --- Instance Variables ---
        # Bank selector (bank is either 0, 1, 2, or 3)
        self._current_bank = 0
        # --- Bias ---
        # Accel bias (m/s^2) - initialise to zero
        self._bias_accel = [0.0, 0.0, 0.0]
        # Gyro bias (rad/s) - initialise to zero
        self._bias_gyro = [0.0, 0.0, 0.0]

        # --- Compute Scale Factors ---
        # Accel scale factor
        self._accel_scale = None
        # Gyro scale factor
        self._gyro_scale = None

        # --- Configure Hardware ---
        # Create and open and store the bus object
        self._bus = smbus2.SMBus(self._i2c_bus)

        # --- Configure accel/gyro ---
        self._configure_gyro()
        self._configure_accel()


    # --- Private Register Access ---
    # Register write of current bank (or bank 0 if you don't prior _set_bank)
    def _write_register(self, reg, value):
        return self._bus.write_byte_data(self._i2c_address, reg, value)
    # Read register of current bank (or bank 0 if you don't prior _set_bank)
    def _read_register(self, reg):
        return self._bus.read_byte_data(self._i2c_address, reg)
    # Set register bank
    def _set_bank(self, bank):
        value = bank << 4
        self._current_bank = bank
        return self._write_register(self._reg_bank_sel, value)
    # Write to a bank's register
    def _write_bank_reg(self, bank, reg, value):
        if self._current_bank != bank:
            self._set_bank(bank)
        self._write_register(reg, value)
    # Read from a bank's register
    def _read_bank_reg(self, bank, reg):
        if self._current_bank != bank:
            self._set_bank(bank)
        return self._read_register(reg)

    # --- Build Config Value Bytes ---
    # Builds GYRO_CONFIG_1 value
    def _pack_gyro_config(self):
        reserved = 0
        cfg = self._gyro_cfg
        value = (
            (reserved & 0b11) << 6 |
            (cfg["dlpf_cfg"] & 0b111) << 3 |
            (cfg["fs_sel"] & 0b11) << 1 |
            (cfg["fchoice"] & 0b1)
        )
        return hex(value)
    # Builds ACCEL_CONFIG value
    def _pack_accel_config(self):
        reserved = 0
        cfg = self._accel_cfg
        value = (
            (reserved & 0b11) << 6 |
            (cfg["dlpf_cfg"] & 0b111) << 3 |
            (cfg["fs_sel"] & 0b11) << 1 |
            (cfg["fchoice"] & 0b1)
        )
        return hex(value)
    
    # --- Methods to Configure Accel / Gyro ---
    # Configure Gyroscope
    def _configure_gyro(self):
        value = self._pack_gyro_config()
        self._write_bank_reg(
            self._sensor_config_bank,
            self._gyro_cfg_address,
            value
        )
    # Configure Accelerometer
    def _configure_accel(self):
        value = self._pack_accel_config()
        self._write_bank_reg(
            self._sensor_config_bank,
            self._accel_cfg_address,
            value
        )

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