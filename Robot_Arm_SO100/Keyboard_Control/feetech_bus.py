# feetech_bus.py
import numpy as np
import scservo_sdk as scs
import config

class FeetechBus:
    def __init__(self, port: str, motors: dict[str, tuple[int, str]]):
        # Store the port and motors dictionary
        self.port = port
        self.motors = motors

        # Initialise internal state
        self.port_handler = None
        self.packet_handler = None
        self.is_connected = False
        self.calibration = None

    def connect(self):
        # Create port and packet handler objects
        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(0)

        # Attempt to open the serial port
        if not self.port_handler.openPort():
            raise IOError(f"Failed to open port '{self.port}'")

        # Set the communication timeout in ms
        self.port_handler.setPacketTimeoutMillis(config.TIMEOUT_MS)
        self.is_connected = True

        # Print connection success
        if config.DEBUG:
            print(f"[DEBUG] Connected to {self.port}")

    # Disconnect from servos
    def disconnect(self):
        # Close the serial port if connected
        if self.is_connected and self.port_handler:
            self.port_handler.closePort()
            self.is_connected = False

            if config.DEBUG:
                print(f"[DEBUG] Disconnected from {self.port}")

    # Set calibration
    def set_calibration(self, calib_dict: dict):
        # Save the calibration data to internal state
        self.calibration = calib_dict

        if config.DEBUG:
            print("[DEBUG] Calibration loaded into motor bus")

    # Disable torque (for calibration)
    def disable_torque(self):
        for motor_name, (motor_id, _) in self.motors.items():
            # Torque Enable = 0
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, 40, 0)
            if config.DEBUG:
                print(f"[DEBUG] Torque disabled on motor '{motor_name}'")

    # Enable torque again
    def enable_torque(self):
        for motor_name, (motor_id, _) in self.motors.items():
            # Torque Enable = 1
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, 40, 1)
            if config.DEBUG:
                print(f"[DEBUG] Torque enabled on motor '{motor_name}'")
    
    # Read position of a motor servo
    def read_position(self, motor_name: str) -> float:
        # Get motor ID and model from config
        motor_id, model = self.motors[motor_name]

        # Read Present_Position (address #56) from motor
        value, result, error = self.packet_handler.read2ByteTxRx(self.port_handler, motor_id, 56)
        if result != scs.COMM_SUCCESS:
            raise IOError(f"Failed to read position from motor '{motor_name}'")

        # Apply calibration to convert raw step value to degrees
        raw = np.int32(value)
        idx = self.calibration["motor_names"].index(motor_name)
        offset = self.calibration["homing_offset"][str(motor_id)]
        drive = self.calibration["drive_mode"][idx]
        resolution = config.DEFAULT_RESOLUTION

        # Adjust raw encoder values to degrees
        adjusted = (raw + offset) / (resolution // 2) * 180.0
        if drive:
            adjusted *= -1
        return adjusted

    # Apply servo angles
    def write_position(self, motor_name: str, target_deg: float) -> bool:
        # Get motor ID and model
        motor_id, model = self.motors[motor_name]
        idx = self.calibration["motor_names"].index(motor_name)
        drive = self.calibration["drive_mode"][idx]
        resolution = config.DEFAULT_RESOLUTION
        calib_mode = self.calibration["calib_mode"][idx]

        # Look up offset using motor ID as string
        offset = self.calibration["homing_offset"][str(motor_id)]

        # Convert calibrated range to degrees
        if calib_mode == "DEGREE":
            start = (self.calibration["start_pos"][idx] + offset) / (resolution // 2) * 180.0
            end = (self.calibration["end_pos"][idx] + offset) / (resolution // 2) * 180.0
            min_deg, max_deg = sorted([start, end])
            if drive:
                min_deg, max_deg = -max_deg, -min_deg
        # Linear mode (not used)
        elif calib_mode == "LINEAR":
            min_deg, max_deg = 0.0, 100.0

        # Reject command if target is outside bounds
        if not (min_deg <= target_deg <= max_deg):
            if config.DEBUG:
                print(f"[DEBUG] {motor_name} angle {target_deg:.1f}° exceeds limits ({min_deg:.1f}° to {max_deg:.1f}°) → clamped.")
            # Clamp servo
            return False

        # Convert degrees to raw steps
        if drive:
            target_deg *= -1
        steps = int(target_deg / 180.0 * (resolution // 2))
        raw_value = steps - offset

        # Write position to motor
        result, _ = self.packet_handler.write2ByteTxRx(self.port_handler, motor_id, 42, raw_value)
        if result != scs.COMM_SUCCESS:
            raise IOError(f"Failed to write position to motor '{motor_name}'")

        # Confirm action
        if config.DEBUG:
            print(f"[DEBUG] Wrote {target_deg:.1f}° to {motor_name} (raw {raw_value})")

        # Write position
        return True
