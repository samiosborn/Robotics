# src/robot/imu/icm20948.py
import math
import yaml
import time
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

    # --- Methods to Read Accel / Gyro ---
    # Read IMU registers
    def _read_sensor_registers(self, registers): 
        return {
        "x_h": self._read_register(registers["x_h"]),
        "x_l": self._read_register(registers["x_l"]),
        "y_h": self._read_register(registers["y_h"]),
        "y_l": self._read_register(registers["y_l"]),
        "z_h": self._read_register(registers["z_h"]),
        "z_l": self._read_register(registers["z_l"]),
        }
    # Combine high and low bits and sign-extend
    def _combine_h_l_as_signed(self, high_byte, low_byte):
        # Combine high and low bytes
        combined = (high_byte <<8 | low_byte)
        # If bit 15 is 1, then it's actually negative
        if combined & 0x8000:
            # Sign-extend
            combined = combined - 0x10000
        return combined
    # Combine all axis H/L pairs in a register dictionary
    def _combine_axis_dict(self, raw_dict_h_l):
        return {
            "x": self._combine_h_l_as_signed(raw_dict_h_l["x_h"], raw_dict_h_l["x_l"]),
            "y": self._combine_h_l_as_signed(raw_dict_h_l["y_h"], raw_dict_h_l["y_l"]),
            "z": self._combine_h_l_as_signed(raw_dict_h_l["z_h"], raw_dict_h_l["z_l"]),
        }
    # Apply sensitivity to dict
    def _sensitivity_for_dict(self, raw_dict, sensitivity): 
        return {
            "x": raw_dict["x"] / sensitivity,
            "y": raw_dict["y"] / sensitivity,
            "z": raw_dict["z"] / sensitivity,
        }
    # Apply scale factor to dict
    def _apply_scale_factor(self, sensitised_dict, scale_factor):
        return {
            "x": sensitised_dict["x"] * scale_factor,
            "y": sensitised_dict["y"] * scale_factor,
            "z": sensitised_dict["z"] * scale_factor,
        }
    # Unbias dict
    def _unbias_dict(self, biased_dict, bias): 
        return {
            "x": biased_dict["x"] - bias[0],
            "y": biased_dict["y"] - bias[1],
            "z": biased_dict["z"] - bias[2],
        }
    # Axis remapping for dict
    def _axis_remapping(self, unbiased_dict):
        # Dict indexing
        mapped = {
            "pitch": unbiased_dict[self._axis_map["pitch"]],
            "roll": unbiased_dict[self._axis_map["roll"]], 
            "yaw": unbiased_dict[self._axis_map["yaw"]],  
        }
        # Pitch inversion
        if self._axis_map.get("invert_pitch", False): 
            mapped["pitch"] = -mapped["pitch"]
        return mapped

    # --- Public API ---
    # Read IMU
    def read(self):
        # Switch to correct bank for sensor values
        self._set_bank(self._sensor_values_bank)
        # Read raw accel and gyro
        accel_raw_separate = self._read_sensor_registers(self._accel_registers)
        gyro_raw_separate = self._read_sensor_registers(self._gyro_registers)
        # Combine high and low bytes into signed 16-bit values
        accel_raw = self._combine_axis_dict(accel_raw_separate)
        gyro_raw  = self._combine_axis_dict(gyro_raw_separate)
        # Convert to SI units
        acc_g = self._sensitivity_for_dict(accel_raw, self._accel_sens)
        acc_ms2 = self._apply_scale_factor(acc_g, self._gravity)
        gyro_dps = self._sensitivity_for_dict(gyro_raw, self._gyro_sens)
        gyro_rad_s = self._apply_scale_factor(gyro_dps, math.pi / 180.0)
        # Subtract bias
        unbiased_acc_ms2 = self._unbias_dict(acc_ms2, self._bias_accel)
        unbiased_gyro_rad_s = self._unbias_dict(gyro_rad_s, self._bias_gyro)
        # Apply axis remapping
        accel = self._axis_remapping(unbiased_acc_ms2)
        gyro = self._axis_remapping(unbiased_gyro_rad_s)
        # Timestamp
        timestamp = time.time()
        # Return dict: "accel", "gyro", "timestamp"
        return {
            "accel": accel,
            "gyro": gyro,
            "timestamp": timestamp,
        }

    # Calibrate IMU
    def calibrate(self, num_calib_samples):
        # Read IMU in SI
        # Average Error
        "self._bias_accel = [bx, by, bz] self._bias_gyro  = [gx, gy, gz]
        "
        # Set Bias for Accel and Gyro
        pass