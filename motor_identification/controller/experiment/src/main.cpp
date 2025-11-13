#include <Arduino.h>
#include <Nidec24H.h>

#include <math.h>


Nidec24H motor(/*brake*/ 27, /*pwm*/ 26, /*dir*/ 25, /*encA*/ 33, /*encB*/ 32, /*freq*/ 20000, /*resolution*/ 8, /*pulsesPerRevolution*/ 100);

void setup()
{
    Serial.begin(115200); // Initialize serial
    motor.begin(); // Initialize the motor controller
    motor.brake(false); // Release the brake
}

void loop()
{
    float angle = 0.0;
    motor.setSpeed(0.1); // Set motor speed to 10% counter-clockwise
    while(angle < 2 * M_PI) { // Rotate until one full revolution
        angle = motor.readAngle();
    }
    motor.brake(true); // Engage the brake
    motor.setSpeed(0.0); // Stop the motor
    Serial.println("Completed one full revolution.");
    while(1) {
        delay(1000); // Keep the program running
    }
}