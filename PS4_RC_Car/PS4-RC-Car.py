
import board
import busio
import adafruit_pca9685
from adafruit_servokit import ServoKit
import evdev
from evdev import list_devices, InputDevice, categorise, ecodes
import logging

# Configuration constants
JOYSTICK_X_MIDDLE_VARIATION = 5
JOYSTICK_X_MIN = 0
JOYSTICK_X_MAX = 255
JOYSTICK_X_MID = 128
JOYSTICK_Y_MIDDLE_VARIATION = 4
S_SET_M = 80
S_SET_DIFF = 40
T_SET_M = 90

# Initialise logging
logging.basicConfig(level=logging.INFO)

def initialise_hat():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        hat = adafruit_pca9685.PCA9685(i2c)
        hat.frequency = 60
        return hat
    except Exception as e:
        logging.error(f"Failed to initialise PCA9685: {e}")
        raise

def initialise_servo_kit():
    try:
        return ServoKit(channels=16)
    except Exception as e:
        logging.error(f"Failed to initialise ServoKit: {e}")
        raise

def find_device(device_name):
    devices = [InputDevice(path) for path in list_devices()]
    for device in devices:
        if device.name == device_name:
            return evdev.InputDevice(device.path)
    logging.error(f"Device {device_name} not found")
    return None

def setup_servo(kit, channel, pulse_width_range):
    kit.servo[channel].set_pulse_width_range(*pulse_width_range)
    kit.servo[channel].angle = S_SET_M

def main():
    hat = initialise_hat()
    kit = initialise_servo_kit()
    dev = find_device("Wireless Controller")
    if not dev:
        return

    setup_servo(kit, 0, (1000, 2000))
    setup_servo(kit, 1, (1000, 2000))

    # Main event loop
    for event in dev.read_loop():
        if event.type == ecodes.EV_ABS:
            handle_event(event, kit)

def handle_event(event, kit):
    if event.code == ecodes.ABS_X:
        handle_x_event(event, kit)
    elif event.code == ecodes.ABS_Y:
        handle_y_event(event, kit)

def handle_x_event(event, kit):
    if abs(event.value - JOYSTICK_X_MID) <= JOYSTICK_X_MIDDLE_VARIATION:
        x_value = JOYSTICK_X_MID
    else:
        x_value = event.value
    kit.servo[0].angle = S_SET_M + S_SET_DIFF * (x_value - JOYSTICK_X_MID) / ((JOYSTICK_X_MAX - JOYSTICK_X_MIN) / 2)

def handle_y_event(event, kit):
    if abs(event.value - JOYSTICK_X_MID) <= JOYSTICK_Y_MIDDLE_VARIATION:
        y_value = JOYSTICK_X_MID
    else:
        y_value = event.value
    kit.servo[1].angle = T_SET_M + S_SET_DIFF * (y_value - JOYSTICK_X_MID) / ((JOYSTICK_X_MAX - JOYSTICK_X_MIN) / 2)

if __name__ == "__main__":
    main()