
#include <Wire.h>

// --- PINS ---

// MPU-6500 IMU
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
constexpr uint8_t MPU_ADDR = 0x68;

// MPU-6500 POWER MANAGEMENT REGISTER
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;

// MPU-6500 IDENTITY REGISTER
constexpr uint8_t REG_WHO_AM_I = 0x75;

// --- CONSTANTS ---

// EXPECTED WHO_AM_I VALUE
constexpr uint8_t EXPECTED_WHO_AM_I = 0x70;

// Exponential Moving Average (EMA) alpha
const float EMA_ALPHA = 0.01;


void setup() {

  // Baud rate (Speed of connection to Serial Monitor)
  Serial.begin(115200);
  delay(50);

  // Note: pinMode not needed for I2C or TM1637
  // Initialise I2C
  Wire.begin();
  
  // Set 400 kHz I2C clock
  Wire.setClock(400000);

  // START transmission with MPU
  Wire.beginTransmission(MPU_ADDR);
  // Set the register pointer to PWR_MGMT_1 register
  Wire.write(REG_PWR_MGMT_1);
  // Selects the PLL clock
  Wire.write(0x01);
  // STOP transmission
  uint8_t err = Wire.endTransmission();
  // Print I2C write error
  if (err) {Serial.print(F("I2C write error: ")); Serial.println(err);}

  // START transmission with MPU
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

}

void loop() {
  // Read acceleration in X/Y/Z axis

  // Scale to units of g

  // Compute roll (rotation around x-axis)

  // Compute pitch (rotation around y-axis)

  // Apply Exponential Moving Average (EMA) Smoothing

  // Print to Serial

}
