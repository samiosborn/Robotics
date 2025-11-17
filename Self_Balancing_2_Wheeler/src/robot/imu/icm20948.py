# src/robot/imu/icm20948.py
import math
import yaml
import smbus2

class ICM20948:
    def __init__(self, yaml_config_path):
        # --- Import Configs ---
        # Import constants
        import src.config.imu_constants as constants
        # Load YAML Config
        with open(yaml_config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # --- Class Variables ---
        # I2C bus number
        self._i2c_bus = cfg["i2c_bus"]
        # I2C address
        self._i2c_address = cfg["i2c_address"]
        # Bank selector register
        self._reg_bank_sel = cfg["reg_bank_sel"]
        # Bank for sensor values
        self._sensor_values_bank = cfg["sensor_values_bank"]
        # Bank for sensor configs
        self._sensor_config_bank = cfg["sensor_config_bank"]
        # Gyro configs
        self._gyro_cfg = cfg["gyro_configs"]
        # Gyro config address
        self._gyro_cfg_address = cfg["gyro_cfg_address"]
        # Gyro value registers
        self._gyro_registers = cfg["gyro_registers"]
        # Accel configs
        self._accel_cfg = cfg["accel_configs"]
        # Accel config address
        self._accel_cfg_address = cfg["accel_cfg_address"]
        # Accel value registers
        self._accel_registers = cfg["accel_registers"]
        # Axis map: dict mapping IMU axes to robot axes
        self._axis_map = cfg["axis_map"]
        # Accel sensitivity
        self._accel_sens = constants.ACCEL_SENSITIVITY_LSB_PER_G[self._accel_cfg["fs_sel"]]
        # Gyro sensitivity
        self._gyro_sens = constants.GYRO_SENSITIVITY_LSB_PER_DPS[self._gyro_cfg["fs_sel"]]
        # Gravity
        self._gravity = constants.GRAVITY_M_S2

        # --- Instance Variables ---
        # Bank selector (bank is either 0, 1, 2, or 3)
        self._current_bank = 0
        # --- Bias ---
        # Accel bias (m/s^2) - initialise to zero
        self._bias_accel = [0.0, 0.0, 0.0]
        # Gyro bias (rad/s) - initialise to zero
        self._bias_gyro = [0.0, 0.0, 0.0]

        # --- Configure Hardware ---
        # Create and open and store the bus object
        self._bus = smbus2.SMBus(self._i2c_bus)
        # Configure accel/gyro
        self._configure_gyro()
        self._configure_accel()


    # --- Private Register Access ---
    # Register write of current bank
    def _write_register(self, reg, value):
        return self._bus.write_byte_data(self._i2c_address, reg, value)
    # Read register of current bank
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
        return value
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
        return value
    
    # --- Methods to Configure Accel / Gyro ---
    # Configure Gyroscope
    def _configure_gyro(self):
        self._write_bank_reg(
            self._sensor_config_bank,
            self._gyro_cfg_address,
            self._pack_gyro_config()
        )
    # Configure Accelerometer
    def _configure_accel(self):
        self._write_bank_reg(
            self._sensor_config_bank,
            self._accel_cfg_address,
            self._pack_accel_config()
        )

    # --- Public API ---
    # Read IMU
    def read(self):
        # Read raw accel and gyro
        self.bank = self._sensor_values_bank
        accel_raw_separate = {
        "x_h": self._read_register(self._accel_registers["x_h"]),
        "x_l": self._read_register(self._accel_registers["x_l"]),
        "y_h": self._read_register(self._accel_registers["y_h"]),
        "y_l": self._read_register(self._accel_registers["y_l"]),
        "z_h": self._read_register(self._accel_registers["z_h"]),
        "z_l": self._read_register(self._accel_registers["z_l"]),
        }
        gyro_raw_separate = {
        "x_h": self._read_register(self._gyro_registers["x_h"]),
        "x_l": self._read_register(self._gyro_registers["x_l"]),
        "y_h": self._read_register(self._gyro_registers["y_h"]),
        "y_l": self._read_register(self._gyro_registers["y_l"]),
        "z_h": self._read_register(self._gyro_registers["z_h"]),
        "z_l": self._read_register(self._accel_registers["z_l"]),
        }
        # Combine high and low bytes
        accel_raw = [
            "ax": (accel_raw_separate["x_h"]) <<8 | (accel_raw_separate["x_l"]), 
            "ay": (accel_raw_separate["y_h"]) <<8 | (accel_raw_separate["y_l"]), 
            "az": (accel_raw_separate["z_h"]) <<8 | (accel_raw_separate["z_l"]),  
        ]
        gyro_raw = {
            "gx": (gyro_raw_separate["x_h"]) <<8 | (gyro_raw_separate["x_l"]), 
            "gy": (gyro_raw_separate["y_h"]) <<8 | (gyro_raw_separate["y_l"]), 
            "gz": (gyro_raw_separate["z_h"]) <<8 | (gyro_raw_separate["z_l"]),  
        }
        # Sign-extend
        # Convert to SI units

        acc_g = raw_accel / self._accel_sens
        acc_ms2 = acc_g * self._gravity

        gyro_dps = raw_gyro / self._gyro_sens
        gyro_rad_s = gyro_dps * math.pi / 180.0
        # Subtract bias
        # Apply axis remapping     
        # Return dict: "accel", "gyro", "timestamp"
        pass
    # Calibrate IMU
    def calibrate(self, num_calib_samples):
        # Read IMU in SI
        # Average Error
        # Set Bias for Accel and Gyro
        pass