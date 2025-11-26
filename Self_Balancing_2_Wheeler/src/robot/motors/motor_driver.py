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
    
    # --- Private Methods --- 
    # Encode float duty command
    def _encode_level(self, duty):
        # Stop
        if duty == 0:
            return 1, 0
        # Clip
        duty = max(-1.0, min(1.0, duty))
        # Determine direction
        direction = 1 if duty >= 0 else 0
        # Scale magniture to -255 to 255
        level = int(duty * 255)
        return direction, level
    # Write motor command
    def _write_motor(self, motor_id, duty):
        # Encode level
        direction, level = self._encode_level(duty)
        # Set drive
        self._scmd.set_drive(motor_id, direction, level)
    
    # --- Public API ---
    # Set duty command
    def set_duty(self, left_duty, right_duty):
        # Write motors
        self._write_motor(self._left_motor_id, left_duty)
        self._write_motor(self._right_motor_id, right_duty)
    # Stop command
    def stop(self):
        # Stop motors
        self._write_motor(self._left_motor_id, 0)
        self._write_motor(self._right_motor_id, 0)
