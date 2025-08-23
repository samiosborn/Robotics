// dB_meter.ino
#include <math.h>

// PINS

// Analogue read from mic board (KY-038)
const uint8_t MIC_PIN = A0;

// ROLLING BUFFER

// Buffer size
const size_t bufferSize = 240;

// Template rolling buffer of last N samples
template<size_t N>
struct RingBuffer{
  // Allocate an array called data
  uint16_t data[N];
  // Index of head of data struct to write next (initialise to 0)
  size_t head = 0;
  // Count of valid data entries (initialise to 0)
  size_t count = 0;

  // Push next entry to the oldest entry on the buffer
  void push(uint16_t v){
    // Set the value at head to the entry
    data[head] = v;
    // Increment head (mod N)
    head = (head + 1) % N;
    // Increment count (until N)
    if (count < N) count ++;
  }

  // Mean of last M samples
  float meanLast(size_t M) const {
    // If no entries, no mean
    if (count == 0) return 0.f;
    // If less than M valid entries, cap M
    if (count < M) M = count;
    // Initialise accumulation for numerator
    uint32_t acc = 0;
    // Iterate over M items
    for (size_t i = 0; i < M; ++i){
      // Head is most recent, so sum backwards
      size_t idx = (head + N - i -1) % N;
      acc += data[idx];
    }
    return (float) acc / M;
  }

  // Get the i-th most recent sample (return success or failure of recall)
  bool getRecent(size_t i, uint16_t &out) const {
    // Return false if sample not long enough
    if (i >= count) return false;
    // Take the sample i back from the head (mod N)
    size_t idx = (head + N - i -1) % N;
    // Write the value into the caller's variable
    out = data[idx];
    // Success
    return true;
  }

  // Root Mean Square of last M items 
  float RMS(size_t M) const {
    // If no entries, no RMS
    if (count == 0) return 0.f;
    // If less than M valid entries, cap M
    if (count < M) M = count;
    // Get the mean of the sample
    float mean = meanLast(M);
    // Initialise the current sample
    uint16_t currentSample = 0;
    // Initialise an accumulator
    float accSquared = 0.0f; 
    // Iterate over M items
    for (size_t i = 0; i < M; ++i){
      // Current value (write to variable)
      getRecent(i, currentSample);
      // Difference between current sample and mean
      float diff = (float) currentSample - mean;
      // Add the square of the diff
      accSquared += diff * diff;
    }
    return sqrtf(accSquared / (float) M);
  }
};

// Create a buffer instance
RingBuffer<bufferSize> micBuf;

// RMS to dB




// SMOOTHING AND DECAY CONSTANTS

// Exponential Moving Average (EMA) smoothing factor
const float EMA_ALPHA = 0.20f;
// Slew-Rate Limiter (SRL) attack rate in dB / s (how fast the signal can rise)
const float SRL_ATTACK_RATE = 30.0f;
// SRL release rate in dB / s (how fast the signal can fall)
const float SRL_RELEASE_RATE = 8.0f; 

// INTERNAL LEVEL STATE VARIABLES

// First set of samples (initialise to TRUE)
static bool first_level = true; 
// Internal EMA state
static float EMA_level = -100.0f;
// SRL output
static float SRL_level = -100.0f;
// Last output time (in ms)
static unsigned long time_last_level_ms = 0; 

// Smoothing Output Level
float updateLevel(float db_now){
  // Capture current time
  unsigned long now = millis();
  
  // EMA Smoothing
  if (first_level){
    // For first time EMA smoothing, just give the raw dB level
    EMA_level = db_now;
    SRL_level = db_now;
    // For the next step, will no longer be the first time
    first_level = false;
  } else {
    // EMA average
    EMA_level = db_now * EMA_ALPHA + EMA_level * (1.0f - EMA_ALPHA);

    // SRL rate (converting ms to s)
    float dt = (now - time_last_level_ms) / 1000.0f;
    // Prevent rounding dt to zero
    if (dt < 0.0005f) dt = 0.0005f;

    // SRL limits
    float maxRise = SRL_ATTACK_RATE * dt;
    float maxFall = SRL_RELEASE_RATE * dt;

    // Cap on change of signal level
    float diff = EMA_level - SRL_level;
    if (diff > maxRise){
      SRL_level += maxRise;
    } else if (diff < -maxFall){
      SRL_level -= maxFall;
    } else {
      SRL_level = EMA_level;
    }
  }
  // Update time of last level out
  time_last_level_ms = now;
  // Return the level
  return SRL_level;
}

// 4-Digit 7-Segment display


void setup() {
  // Baudrate of 9600
  Serial.begin(9600);

  // Note: No pinMode needed for analogue inputs

}

void loop() {
  // Current reading from mic
  int currentMicReading = analogRead(MIC_PIN);

  // Push into rolling buffer
  micBuf.push((uint16_t)currentMicReading);

  // Print to Serial Monitor
  Serial.println(currentMicReading);

}
