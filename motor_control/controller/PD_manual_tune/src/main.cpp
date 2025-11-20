#include <Arduino.h>
#include <Nidec24H.h>

// Sampling interval in milliseconds
#define SAMPLE_INTERVAL_MS 10
#define INPUT_CHANGE_PERIOD_SAMPLES 100 // Change input every 100 samples

// Motor instance
Nidec24H motor(27, 26, 25, 33, 32, 20000, 8, 100);

// State variables
unsigned long lastSampleTime = 0;
double lastAngle = 0.0;
double lastVelocity = 0.0;
float currentSpeed = 0.0;

// Sample counter for input change timing
unsigned long sampleCounter = 0;

// Data packet structure for binary transmission
struct DataPacket {
    uint32_t timestamp;    // milliseconds
    float angle;           // radians
    float velocity;        // rad/s
    float acceleration;    // rad/s^2
    float speedCommand;    // -1.0 to 1.0
} __attribute__((packed));

void setup() {
    Serial.begin(115200);
    
    motor.begin();
    
    // Wait for serial connection
    delay(1000);
    
    // Send header marker for host to identify start of stream
    Serial.println("START");
    Serial.flush(); // Ensure START message is fully sent before binary data
    
    lastSampleTime = millis();

    motor.brake(false); // Release brake
}

void loop() {
    unsigned long currentTime = millis();
    
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        // Calculate actual time delta for accurate derivatives
        float dt = (currentTime - lastSampleTime) / 1000.0; // convert to seconds
        
        // Read current angle
        double currentAngle = motor.readAngle();
        
        // Compute angular velocity (first derivative)
        double currentVelocity = (currentAngle - lastAngle) / dt;
        
        // Compute angular acceleration (second derivative)
        double currentAcceleration = (currentVelocity - lastVelocity) / dt;
        
        // Update speed command every INPUT_CHANGE_PERIOD_SAMPLES
        if (sampleCounter % INPUT_CHANGE_PERIOD_SAMPLES == 0) {
            // Generate random speed command in range [-0.5, 0.5]
            // esp_random() returns uint32_t, normalize to [-0.5, 0.5]
            currentSpeed = ((float)esp_random() / (float)UINT32_MAX) * 1.0f - 0.5f;
            motor.setSpeed(currentSpeed);
        }
        
        // Prepare data packet
        DataPacket packet;
        packet.timestamp = currentTime;
        packet.angle = (float)currentAngle;
        packet.velocity = (float)currentVelocity;
        packet.acceleration = (float)currentAcceleration;
        packet.speedCommand = currentSpeed;
        
        // Send binary packet
        Serial.write((uint8_t*)&packet, sizeof(DataPacket));
        
        // Update state variables
        lastAngle = currentAngle;
        lastVelocity = currentVelocity;
        lastSampleTime = currentTime;

        // Increment sample counter
        sampleCounter ++;
    }
}
