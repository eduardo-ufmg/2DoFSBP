#include <Arduino.h>
#include <Comms.h>
#include <Nidec24H.h>

#include "bootloader_random.h"

const unsigned int testDataLength = 4096;
const unsigned int samplePeriodMs = 10;
const unsigned int inputChangeTimeMs = 100;
const unsigned int inputArrayLength = testDataLength / (inputChangeTimeMs / samplePeriodMs);
float inputValues[inputArrayLength];

Nidec24H motor(27, 26, 25, 33, 32, 20000, 8, 100);

ResultCode runMotorTest();
ResultCode sendTestData();

typedef struct
{
    float input[testDataLength];
    float angle[testDataLength];
    float timeS[testDataLength];
} TestData;

TestData testData;
ResultCode testResult = RESULT_ERROR;

void setup()
{

    // Re-enable random source temporarily
    // It is disabled after the bootloader runs,
    // but keeping it enabled some more helps to
    // get better entropy for esp_random()
    bootloader_random_enable();

    Serial.begin(115200);
    motor.begin();

    pinMode(LED_BUILTIN, OUTPUT);

    // Check connection with host
    if (connectionCheck() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Wait for start command from host
    if (waitForStartCommand() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Acknowledge start command
    if (ackStartCommand() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Run motor test
    if (runMotorTest() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Send success message to host
    if (sendSuccessMessage() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Wait for data request from host
    if (waitForDataRequest() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Acknowledge data request
    if (ackDataRequest() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    // Send test data to host
    if (sendTestData() != RESULT_OK) {
        testResult = RESULT_ERROR;
        return;
    }

    testResult = RESULT_OK;

    // Disable random source after enough entropy is gathered
    bootloader_random_disable();
}

void loop()
{

    int ledHalfPeriodMs = (testResult == RESULT_OK) ? 200 : 1000;

    digitalWrite(LED_BUILTIN, HIGH);
    delay(ledHalfPeriodMs);
    digitalWrite(LED_BUILTIN, LOW);
    delay(ledHalfPeriodMs);
}

ResultCode runMotorTest()
{

    // Pre-generate random input values
    for (unsigned int i = 0; i < inputArrayLength; i++) {
        inputValues[i] = (static_cast<float>(esp_random()) / UINT32_MAX) * 0.2f -
                         0.1f; // Random value between -0.1 and +0.1
    }

    float inputValue = 0.0f;
    unsigned int inputIndex = 1; // Start from second input since first is already set to 0.0f

    motor.brake(false);
    motor.setSpeed(inputValue);

    delay(1000); // Allow some time for motor to start

    unsigned int testStartTimeMs = millis();
    unsigned int inputChangeLastTimeMs = testStartTimeMs,
                 inputChangeCurrentTimeMs = testStartTimeMs;
    unsigned int sampleLastTimeMs = testStartTimeMs, sampleCurrentTimeMs = testStartTimeMs;

    motor.resetEncoder();

    for (unsigned int i = 0; i < testDataLength; i++) {
        testData.input[i] = inputValue;
        testData.angle[i] = motor.readAngle();
        testData.timeS[i] = (millis() - testStartTimeMs) / 1000.0f;

        inputChangeCurrentTimeMs = millis(); // Update current time for input change check
        if (inputChangeCurrentTimeMs - inputChangeLastTimeMs >=
            inputChangeTimeMs) {                              // Time to change input
            inputValue = inputValues[inputIndex];             // Get next input value
            motor.setSpeed(inputValue);                       // Apply new input to motor
            inputIndex++;                                     // Increment input index
            inputChangeLastTimeMs = inputChangeCurrentTimeMs; // Update last input change time
        }

        sampleCurrentTimeMs = millis(); // Update current time for sampling
        while (sampleCurrentTimeMs - sampleLastTimeMs < samplePeriodMs) {
            // Wait until it's time for the next sample
            sampleCurrentTimeMs = millis();
            yield(); // To keep watchdog happy
        }
        sampleLastTimeMs = sampleCurrentTimeMs; // Update last sample time
    }

    motor.setSpeed(0.0f);
    motor.brake(true);

    return RESULT_OK;
}

ResultCode sendTestData()
{
    Serial.write(DEVICE_DATA_STREAM_START);

    Serial.flush();

    Serial.write((uint8_t *)testData.input, sizeof(testData.input));
    Serial.write((uint8_t *)testData.angle, sizeof(testData.angle));
    Serial.write((uint8_t *)testData.timeS, sizeof(testData.timeS));

    Serial.flush();

    Serial.write(DEVICE_DATA_STREAM_END);
    return RESULT_OK;
}
