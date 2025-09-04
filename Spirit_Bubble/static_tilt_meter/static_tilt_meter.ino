#include <math.h>
#include <Wire.h>
#include <TM1637Display.h>

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
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
// ACCEL CONFIG REGISTER
const uint8_t REG_ACCEL_CONFIG = 0x1C;


// --- CONSTANTS ---

// 1 G PER ACCEL RAW OUT
const float ACCEL_LSB_PER_G = 16384.0f;

// DEGREE PER 1 RAD
const float DEGREE_PER_RAD = 180.0f / PI;

// EXPECTED WHO_AM_I VALUE
const uint8_t EXPECTED_WHO_AM_I = 0x70;

// EXPONENTIAL MOVING AVERAGE (EMA) ALPHA
const float EMA_ALPHA = 0.1;

// --- GLOBAL VARIABLES ---

// EMA PITCH AND ROLL
float pitchEMA = NAN, rollEMA = NAN;

// COORDINATE BIAS (RAW)
float ax_bias = 0, ay_bias = 0, az_bias = 0;

// AXIS SIGNS
int8_t SX = +1, SY = +1, SZ = +1;


// --- DISPLAY ---

// Segment pattern for minus sign (segment G)
static const uint8_t SEG_MINUS = 0x40;

// Initialise Display
TM1637Display display(PIN_TM1637_CLK, PIN_TM1637_DIO);

// --- Read N bytes from starting register ---
bool readBytes(uint8_t startReg, uint8_t* buffer, uint8_t len){
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
    buffer[i] = Wire.read();
  }
  // If everything works, return true
  return true;
}


// --- Read raw accel (16-bit signed per axis) ---
bool readAccelRaw(int16_t& ax, int16_t& ay, int16_t& az){
  // Initialise an array of 6
  uint8_t b[6];
  // Bail if reading fails
  if (readBytes(REG_ACCEL_XOUT_H, b, 6) == false) return false;
  // IMU returns 16-bit values for the 3 axis (in big-endian order)
  ax = (int16_t)((b[0] << 8) | b[1]);
  ay = (int16_t)((b[2] << 8) | b[3]);
  az = (int16_t)((b[4] << 8) | b[5]);
  // If we got so far, it works
  return true;
}


// --- Return acceleration in g and remove bias ---
bool readAccelG(float& ax_g, float& ay_g, float& az_g){
  // Initialise raw reading variables
  int16_t ax_r, ay_r, az_r;
  // Read raw accel & If read doesn't work, return FALSE
  if (readAccelRaw(ax_r, ay_r, az_r) == false) return false;
  // Remove bias and convert to g
  float axg = (float(ax_r) - ax_bias) / ACCEL_LSB_PER_G; 
  float ayg = (float(ay_r) - ay_bias) / ACCEL_LSB_PER_G; 
  float azg = (float(az_r) - az_bias) / ACCEL_LSB_PER_G; 
  // Correct if upside down
  ax_g = SX * axg;
  ay_g = SY * ayg;
  az_g = SZ * azg;
  // If so far works, return TRUE
  return true;
} 


// --- Calculate PITCH/ROLL (in radians) from accel (in g) ---
void accelToPitchRoll(float ax, float ay, float az, float& pitch_rad, float& roll_rad){
  // Roll (about x-axis)
  roll_rad = atan2f(ay, az);
  // Pitch (about y-axis)
  pitch_rad = atan2f(-ax, sqrtf(ay*ay + az*az));
}


// --- Convert Radians to Degrees ---
inline float rad2deg(float r){
  return r * DEGREE_PER_RAD;
}


// --- Update EMA Smoothing ---
inline float updateEMA (float ema_prev, float new_reading, float alpha){
  // For the first time, just output the first reading
  if (isnan(ema_prev)) return new_reading;
  // Otherwise, update the average
  return alpha * new_reading + (1.0f - alpha) * ema_prev;
}


// --- Calibration ---
void calibrateAccelOffsets(){
  // Ask user to level IMU
  Serial.println(F("Level the IMU, you have 3 seconds!"));
  // Wait 3 seconds
  delay(3000);
  // Print starting calibration
  Serial.println(F("Calibration has started."));
  
  // Average over N steps
  const uint16_t N = 300;
  // Initialise sum of raw readings
  long sum_x = 0, sum_y = 0, sum_z = 0;
  // Loop and sum
  for (uint16_t i = 0; i < N; ++i){
    // Initialise raw readings
    int16_t axr, ayr, azr;
    // If reading IMU works
    if (readAccelRaw(axr, ayr, azr)){
      sum_x += axr, sum_y += ayr, sum_z += azr;
    }
    // Short delay before next reading
    delay(2);
  }

  // Average readings
  float m_x = sum_x / (float)N;
  float m_y = sum_y / (float)N;
  float m_z = sum_z / (float)N;

  // Set bias as mean raw value for x and y
  ax_bias = m_x;
  ay_bias = m_y;

  // If Z points down at rest, flip it
  SZ = (m_z < 0) ? -1 : +1;

  // Want at +1 g at rest
  az_bias = m_z - SZ * ACCEL_LSB_PER_G;

  // Report bias set
  Serial.println(F("Calibration is complete."));

  // Wait 3 seconds
  delay(3000);

}


// --- Display Angle on TM1637 ---


// --- DEBUG ---
void printAccelDebug() {
  int16_t axr, ayr, azr;
  if (!readAccelRaw(axr, ayr, azr)) { Serial.println(F("raw fail")); return; }
  float ax_g = (float(axr) - ax_bias) / ACCEL_LSB_PER_G;
  float ay_g = (float(ayr) - ay_bias) / ACCEL_LSB_PER_G;
  float az_g = (float(azr) - az_bias) / ACCEL_LSB_PER_G;
  float n = sqrtf(ax_g*ax_g + ay_g*ay_g + az_g*az_g);
  Serial.print(F("RAW ")); Serial.print(axr); Serial.print(' ');
  Serial.print(ayr); Serial.print(' '); Serial.print(azr);
  Serial.print(F("  |  g ")); Serial.print(ax_g,3); Serial.print(' ');
  Serial.print(ay_g,3); Serial.print(' '); Serial.print(az_g,3);
  Serial.print(F("  | |a|=")); Serial.println(n,3);
}

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

  // Calibrate the IMU
  calibrateAccelOffsets();

}

void loop() {
  // Read acceleration in g (bias removed)
  float ax_g, ay_g, az_g;
  readAccelG(ax_g, ay_g, az_g);

  // Compute pitch and roll (in radians)
  float pitch_rad, roll_rad;
  accelToPitchRoll(ax_g, ay_g, az_g, pitch_rad, roll_rad); 

  // Get pitch and roll in degrees
  float pitch_deg, roll_deg;
  pitch_deg = rad2deg(pitch_rad);
  roll_deg = rad2deg(roll_rad);

  // Apply Exponential Moving Average (EMA) Smoothing
  pitchEMA = updateEMA(pitchEMA, pitch_deg, EMA_ALPHA);
  rollEMA = updateEMA(rollEMA, roll_deg, EMA_ALPHA);

  // Print to Serial
  Serial.print(F("Pitch(deg): "));
  Serial.print(pitchEMA, 1);
  Serial.print(F(" & Roll (deg): "));
  Serial.println(rollEMA, 1);

  // Debug
  //printAccelDebug();

  // 50Hz
  delay(20);

}
