// PINS

// pin for the switch (input)
const uint8_t SWITCH_PIN = 3;

// pin for the motor PWM (output)
const uint8_t MOTOR_PWM_PIN = 5;


// MOTOR CONTROL

// Motor active flag
bool motorActive = false;

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


// SWITCH TIMING AND DEBOUNCE

// Scheduled off at time
unsigned long motorOffAt = 0;

// Debounce time in milliseconds
const unsigned long DEBOUNCE_MS = 25;

// Last switch reading (default to HIGH)
int lastSwitchReading = HIGH;

// Stable switch reading (default to HIGH)
int stableSwitchReading = HIGH;

// Last switch change time (start at 0)
unsigned long lastChangeTime = 0;


// SETUP

void setup() {
  // Seed RNG from floating analogue pin 0
  randomSeed(analogRead(A0));

  // CONFIGURE PINS

  // Pin to switch (connected to GND) - pulled-up
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  // Pin to transistor base (output)
  pinMode(MOTOR_PWM_PIN, OUTPUT);

  // Start with motor OFF
  analogWrite(MOTOR_PWM_PIN, 0); 

  // INITIALISE TIMING
  lastChangeTime = millis();
  lastSwitchReading = digitalRead(SWITCH_PIN);
  stableSwitchReading = lastSwitchReading;  
}

// LOOP

void loop() {
  // Take time now
  unsigned long now = millis();

  // Read current switch
  int currentSwitchReading = digitalRead(SWITCH_PIN);

  // If the current switch reading is different to before, take the new time and reading
  if (currentSwitchReading != lastSwitchReading){
    lastChangeTime = now;
    lastSwitchReading = currentSwitchReading;
  }

  // If the switch reading has not changed for a while, accept it 
  if ((now - lastChangeTime) >= DEBOUNCE_MS){
    
    // Only accept it if it's different to the existing stable switch reading
    if (lastSwitchReading != stableSwitchReading){
      stableSwitchReading = lastSwitchReading;

      // Turn the motor on if the stable switch reading is LOW (and the motor is OFF already)
      if (stableSwitchReading == LOW && motorActive == false){
        // Turn on the motor
        analogWrite(MOTOR_PWM_PIN, DUTY_80PC);
        
        // Set the motor active flag to true
        motorActive = true;

        // Time motor should turn off
        motorOffAt = now + randomOnTime();
      }
    }   
  }

  // Turn off the motor if sufficient time has passed (and motor is ON currently)
  if (now > motorOffAt && motorActive == true){
    // Turn off the motor
    analogWrite(MOTOR_PWM_PIN, 0);
    
    // Set the active motor flag to false
    motorActive = false;
  }

}
