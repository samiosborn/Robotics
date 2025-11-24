# src/robot/motors/motor_driver.py
import yaml
import time
import qwiic_scmd

class MotorDriver:
    def __init__(self, motor_yaml_config_path):
        # Import YAML config
        with open(motor_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)["driver"]
        
        # --- Load from YAML ---
        # I2C address for the SCMD master chip
        self._i2c_address = cfg["i2c_address"]
        # Left Motor ID
        self._left_motor_id = cfg["left_motor_id"]
        # Right Motor ID
        self._right_motor_id = cfg["right_motor_id"]
        # Maximum duty command
        self._max_duty_float = cfg["max_duty_float"]
        # Max hardware level 
        self._max_level_hw = cfg["max_level_hardware"]
        
        # --- I2C SCMD ---
        # Init SCMD
        self._scmd = qwiic_scmd.QwiicScmd(self._i2c_address)
        # Begin (wake)
        self._scmd.begin()
        # Wait until the SCMD firmware is ready (i.e. finished internal enumerating) 
        while not self._scmd.ready():
            # Wait for 10ms
            time.sleep(0.01)
        # Enable H-bridge outputs
        self._scmd.enable()
    
