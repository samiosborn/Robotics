# src/robot/motors/encoder.py
import yaml
import math
import time
from collections import deque
import RPi.GPIO as GPIO

class Encoder(): 
    def __init__(self, motor_yaml_config_path):
        # Import YAML config
        with open(motor_yaml_config_path, 'r') as f: 
            cfg = yaml.safe_load(f)
        
        # --- Load from YAML ---
        # Pulses per revolution (per channel)
        self._encoder_ppr_channel = cfg["encoder_ppr_channel"]
        # Motor shaft turns per wheel shaft turns
        self._gear_ratio = cfg["gear_ratio"]
        # Wheel diameter (m)
        self._wheel_diameter_m = cfg["wheel_diameter_m"]
        # Left wheel channel A pin
        self._left_a_pin = cfg["motor_encoder_pins"]["left"]["channel_a"]
        # Left wheel channel B pin
        self._left_b_pin = cfg["motor_encoder_pins"]["left"]["channel_b"]
        # Right wheel channel A pin
        self._right_a_pin = cfg["motor_encoder_pins"]["right"]["channel_a"]
        # Right wheel channel B pin
        self._right_b_pin = cfg["motor_encoder_pins"]["right"]["channel_b"]
        # Length tick timestamp buffer
        self._length_tick_buffer = cfg["length_tick_buffer"]
        # Velocity low-pass filter coefficient alpha
        self._alpha = cfg["alpha"]

        # --- State Variables ---
        # Transition lookup table
        self._transition_table = {
            (0,1): +1,
            (1,3): +1,
            (3,2): +1,
            (2,0): +1,
            (0,2): -1,
            (2,3): -1,
            (3,1): -1,
            (1,0): -1,
        }

        # --- Derivied Constants ---
        # Edge counts per revolution (A & B channel, rising and falling edges)
        self._encoder_cpr_raw = 4 * self._encoder_ppr_channel
        # Wheel radius (m)
        self._wheel_radius_m = self._wheel_diameter_m / 2.0
        # Wheel counts per revolution
        self._wheel_cpr = self._encoder_cpr_raw * self._gear_ratio
        # Radians per count
        self._rad_per_count = 2 * math.pi / self._wheel_cpr

        # --- Configure Encoder Pins ---
        # Stop warnings
        GPIO.setwarnings(False)
        # Set mode to BCM numbering
        GPIO.setmode(GPIO.BCM)
        # Set to in mode, apply pull-ups
        GPIO.setup(self._left_a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self._left_b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self._right_a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self._right_b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        # Event detection
        GPIO.add_event_detect(self._left_a_pin, GPIO.BOTH, callback=self._left_callback, bouncetime=0)
        GPIO.add_event_detect(self._left_b_pin, GPIO.BOTH, callback=self._left_callback, bouncetime=0)
        GPIO.add_event_detect(self._right_a_pin, GPIO.BOTH, callback=self._right_callback, bouncetime=0)
        GPIO.add_event_detect(self._right_b_pin, GPIO.BOTH, callback=self._right_callback, bouncetime=0)

        # --- Instance Variables ---
        # Count
        self._count_left = 0
        self._count_right = 0
        # Previous state
        self._prev_state_left = self._read_current_levels("left")
        self._prev_state_right = self._read_current_levels("right")
        # Tick timestamp buffer
        self._tick_times_left = deque(maxlen=self._length_tick_buffer)
        self._tick_times_right = deque(maxlen=self._length_tick_buffer)
        # Previous velocity
        self._velocity_left_rad_s = 0.0
        self._velocity_right_rad_s = 0.0
        self._velocity_left_m_s = 0.0
        self._velocity_right_m_s = 0.0

    # --- Private Methods ---
    # Read current levels 
    def _read_current_levels(self, wheel_side): 
        # Read GPIO
        if wheel_side == "left": 
            a = GPIO.input(self._left_a_pin)
            b = GPIO.input(self._left_b_pin)
        elif wheel_side == "right":
            a = GPIO.input(self._right_a_pin)
            b = GPIO.input(self._right_b_pin)
        # Current state
        current_state = (a << 1) | b
        return current_state
    # Form quadrature state
    def _quadrature_state(self, current_state, wheel_side):
        if wheel_side == "left":
            previous_state = self._prev_state_left
        elif wheel_side == "right":
            previous_state = self._prev_state_right
        return current_state, previous_state
    # Check transition direction
    def _transition_direction(self, current_state, previous_state, wheel_side):
        delta = self._transition_table.get((previous_state, current_state), 0)
        # Update previous state
        if wheel_side == "left":
            self._prev_state_left = current_state
        elif wheel_side == "right":
            self._prev_state_right = current_state
        return delta
    # Update count
    def _update_count(self, delta, wheel_side):
        if wheel_side == "left": 
            self._count_left += delta
        elif wheel_side == "right":
            self._count_right += delta
        pass
    # Append to tick timestamp buffer
    def _append_tick_times(self, wheel_side):
        timestamp = time.monotonic()
        if wheel_side == "left":
            self._tick_times_left.append(timestamp)
        elif wheel_side == "right":
            self._tick_times_right.append(timestamp)
        pass
    # Compute velocity
    def _compute_velocity(self, tick_buffer, previous_velocity_rad_s):
        # Number of timestamps
        N = len(tick_buffer)
        # Buffer too small
        if N < 2:
            return 0, 0
        # Compute dt
        dt = tick_buffer[N-1] - tick_buffer[0] 
        # Counts per second
        counts_per_sec = (N-1) / dt
        # Angular velocity (rad/s)
        current_velocity_rad_s = counts_per_sec * self._rad_per_count 
        # Low-pass filter
        vel_rad_s = self._alpha * current_velocity_rad_s + (1 - self._alpha) * previous_velocity_rad_s
        # Linear velocity (m/s)
        vel_m_s = vel_rad_s * self._wheel_radius_m
        return vel_rad_s, vel_m_s
    # Update velocity
    def _update_velocity(self, wheel_side):
        if wheel_side == "left":
            vel_rad_s, vel_m_s = self._compute_velocity(
                self._tick_times_left, 
                self._velocity_left_rad_s)
            # Update velocity
            self._velocity_left_rad_s = vel_rad_s
            self._velocity_left_m_s = vel_m_s
        elif wheel_side == "right":
            vel_rad_s, vel_m_s = self._compute_velocity(
                self._tick_times_right, 
                self._velocity_right_rad_s)
            # Update velocity
            self._velocity_right_rad_s = vel_rad_s
            self._velocity_right_m_s = vel_m_s
        pass
    # Event callback
    def _event_callback(self, wheel_side):
        # Read current levels
        current_state = self._read_current_levels(wheel_side)
        # Form quadrature state
        current_state, previous_state = self._quadrature_state(current_state, wheel_side)
        # Check transition direction
        delta = self._transition_direction(current_state, previous_state, wheel_side)
        # Update count
        self._update_count(delta, wheel_side)
        # Append to tick timestamp buffer
        self._append_tick_times(wheel_side) 
        # Update velocity
        self._update_velocity(wheel_side)
        pass
    # Left Callback
    def _left_callback(self, channel):
        self._event_callback("left")
    # Right callback
    def _right_callback(self, channel):
        self._event_callback("right")

    # --- Public API ---
    # Counts
    def get_counts(self):
        return self._count_left, self._count_right
    # Get angular velocity (rad/s)
    def get_angular_velocity(self):
        return self._velocity_left_rad_s, self._velocity_right_rad_s
    # Get linear velocity (m/s)
    def get_linear_velocity(self):
        return self._velocity_left_m_s, self._velocity_right_m_s
    # Reset counts
    def reset_counts(self):
        self._count_left = 0
        self._count_right = 0
    # Reset velocities
    def reset_velocities(self):
        self._velocity_left_rad_s = 0.0
        self._velocity_left_m_s = 0.0
        self._velocity_right_rad_s = 0.0
        self._velocity_right_m_s = 0.0
    # Cleanup GPIO
    def cleanup(self):
        GPIO.remove_event_detect(self._left_a_pin)
        GPIO.remove_event_detect(self._left_b_pin)
        GPIO.remove_event_detect(self._right_a_pin)
        GPIO.remove_event_detect(self._right_b_pin)

