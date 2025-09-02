
#include <Wire.h>

// --- PINS ---

// MPU-6500 IMU (VIA I2C)
// SERIAL CLOCK LINE (SCL)
const uint8_t PIN_SCL = 21;
// SERIAL DATA LINE (SDA)
const uint8_t PIN_SDA = 20;

// TM1637 4-DIGIT DISPLAY
// CLOCK (CLK)
const uint8_t PIN_TM1637_CLK = 6;
// DATA I/O (DIO)
const uint8_t PIN_TM1637_DIO = 7;


// --- ADDRESS AND REGISTERS ---

// MPU-6500 I2C DEVICE ADDRESS
const uint8_t MPU_ADDR = 0x68;
// MPU-6500 POWER MANAGEMENT REGISTER
const uint8_t REG_PWR_MGMT_1 = 0x6B;
// MPU-6500 IDENTITY REGISTER
const uint8_t REG_WHO_AM_I = 0x75;

// ACCEL OUT REGISTER (STARTING WITH X_H)
const uint8_t REG_ACCEL_OUT_X_H = 0x3B;
// ACCEL CONFIG REGISTER
const uint8_t REG_ACCEL_CONFIG = 0x1C;


// --- CONSTANTS ---

// 1 G PER ACCEL RAW OUT
const float ACCEL_LSB_PER_G = 16384.0f;

// EXPECTED WHO_AM_I VALUE
const uint8_t EXPECTED_WHO_AM_I = 0x70;

// EXPONENTIAL MOVING AVERAGE (EMA) ALPHA
const float EMA_ALPHA = 0.01;

// EMA PITCH AND ROLL
float pitchEMA = NAN, rollEMA = NAN;

// COORDINATE BIAS
float ax_bias = 0, ay_bias = 0, az_bias = 0;


// --- Read N bytes from starting register ---
bool readBytes(uint8_t startReg, uint8_t* buf, uint8_t len){
  // Open I2C write transaction to MPU
  Wire.beginTransmission(MPU_ADDR);
  // Send the starting register address
  Wire.write(startReg);
  // Repeated START (if this doesn't work, bail)
  if (Wire.endTransmission(false) != 0) return false;
  // Request len bytes from MPU address --> Return number of bytes
  uint8_t got = Wire.requestFrom(MPU_ADDR, (uint8_t)len);
  // Exit if not the right number of bytes read
  if (got != len) return false;
  // Save the bytes read into the caller's buffer
  for (uint8_t i=0; i<len; ++i){
    buf[i] = Wire.read();
  }
  // If everything works, return true
  return true;
}

// --- Read raw accel (16-bit signed per axis) ---


// --- Convert raw accel read to g and remove bias ---


// --- Calculate PITCH/ROLL from accel (in radians) ---


// --- Convert Radians to Degrees ---


// --- Update EMA Smoothing ---

void setup() {

  // Baud rate (Speed of connection to Serial Monitor)
  Serial.begin(115200);
  delay(50);

  // Note: pinMode not needed for I2C or TM1637
  // Initialise I2C
  Wire.begin();
  
  // Set 400 kHz I2C clock
  Wire.setClock(400000);

  // Start the MPU (START transmission)
  Wire.beginTransmission(MPU_ADDR);
  // Set the register pointer to PWR_MGMT_1 register
  Wire.write(REG_PWR_MGMT_1);
  // Selects the PLL clock
  Wire.write(0x01);
  // STOP transmission
  uint8_t err = Wire.endTransmission();
  // Print I2C write error
  if (err) {Serial.print(F("I2C write error: ")); Serial.println(err);}

  // Double check MPU register (START transmission with MPU)
  Wire.beginTransmission(MPU_ADDR);
  // Write to WHO_AM_I register
  Wire.write(REG_WHO_AM_I);
  // Repeated START (no STOP transmission)
  Wire.endTransmission(false);
  // Request to read 1 byte from the MPU
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  // Should equal EXPECTED_WHO_AM_I
  if (Wire.available()) {
    uint8_t who = Wire.read();
    if (who == EXPECTED_WHO_AM_I) {
      Serial.println(F("MPU-6500 OK"));
    } else {
      Serial.print(F("WHO_AM_I mismatch. Got 0x"));
      Serial.print(who, HEX);
      Serial.print(F(", expected 0x"));
      Serial.println(EXPECTED_WHO_AM_I, HEX);
      }
    } else { 
      Serial.println(F("WHO_AM_I read failed"));
  }

  // Set acceleration full-scale to +- 2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_CONFIG);
  Wire.write(0x00);
  Wire.endTransmission();


}

void loop() {
  // Read acceleration in X/Y/Z axis

  // Scale to units of g

  // Compute roll (rotation around x-axis)

  // Compute pitch (rotation around y-axis)

  // Apply Exponential Moving Average (EMA) Smoothing

  // Print to Serial

}
