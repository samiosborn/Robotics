#include <Arduino.h>

// pin for the switch (input)
const uint8_t SWITCH_PIN = 3;

// pin for the motor PWM (output)
const uint8_t MOTOR_PWM_PIN = 5;

// Debounce time in milliseconds
const unsigned long DEBOUNCE_MS = 25;

// Duty cycle of 80%
const uint8_t DUTY_80PC = 204;

// Motor on time (minimum)
const unsigned long ON_MIN_MS = 2000;

// Motor on time (maximum)
const unsigned long ON_MAX_MS = 3000;

// Random on time generator for motor 
unsigned long randomOnTime(){
  return (unsigned long) random((long)ON_MIN_MS, (long)ON_MAX_MS + 1L);
}

void setup() {
  // Seed RNG from floating analogue pin 0
  randomSeed(analogRead(A0));

  // Pin to switch (connected to GND) - pulled-up
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  // Pin to transistor base (output)
  pinMode(MOTOR_PWM_PIN, OUTPUT);

  // Start with motor OFF
  analogWrite(MOTOR_PWM_PIN, 0); 

}

void loop() {

}
