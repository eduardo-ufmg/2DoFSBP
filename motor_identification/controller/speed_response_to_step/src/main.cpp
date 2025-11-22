#include <Arduino.h>
#include <Nidec24H.h>
#include <Comms.h>
#include "bootloader_random.h"

const unsigned int testDataLength = 2000;
const unsigned int samplePeriodMs = 10;
const float stepInput = 0.1f;        // Commanded speed after step
const unsigned int preStepHoldMs = 5000; // 5s at initial small speed
const float initialInput = -0.1f;

typedef struct {
    float input[testDataLength];
    float angle[testDataLength];
    // removed speed[]; host computes speed offline
} StepData;

StepData dataBuf;
ResultCode testResult = RESULT_ERROR;
Nidec24H motor(27, 26, 25, 33, 32, 20000, 8, 100);

ResultCode runStepTest();
ResultCode sendData();

void setup() {
    bootloader_random_enable();
    Serial.begin(115200);
    motor.begin();
    pinMode(LED_BUILTIN, OUTPUT);

    if (connectionCheck() != RESULT_OK) return;
    if (waitForStartCommand() != RESULT_OK) return;
    if (ackStartCommand() != RESULT_OK) return;
    if (runStepTest() != RESULT_OK) return;
    if (sendSuccessMessage() != RESULT_OK) return;
    if (waitForDataRequest() != RESULT_OK) return;
    if (ackDataRequest() != RESULT_OK) return;
    if (sendData() != RESULT_OK) return;

    testResult = RESULT_OK;
    bootloader_random_disable();
}

void loop() {
    int half = (testResult == RESULT_OK) ? 300 : 1000;
    digitalWrite(LED_BUILTIN, HIGH); delay(half);
    digitalWrite(LED_BUILTIN, LOW);  delay(half);
}

ResultCode runStepTest() {
    motor.brake(false);
    motor.setSpeed(initialInput);
    motor.resetEncoder();

    unsigned long startMs = millis();
    unsigned long lastSampleMs = startMs;
    bool stepApplied = false;

    for (unsigned int i = 0; i < testDataLength; i++) {

        unsigned long nowMs = millis();
        if (!stepApplied && (nowMs - startMs) >= preStepHoldMs) {
            motor.setSpeed(stepInput);
            stepApplied = true;
        }

        dataBuf.input[i] = stepApplied ? stepInput : initialInput;
        dataBuf.angle[i] = motor.readAngle();

        while ((millis() - lastSampleMs) < samplePeriodMs) {
            yield();
        }
        lastSampleMs = millis();
    }

    motor.setSpeed(0.0f);
    motor.brake(true);
    return RESULT_OK;
}

ResultCode sendData() {
    Serial.write(DEVICE_DATA_STREAM_START);
    Serial.flush();
    Serial.write((uint8_t*)dataBuf.input, sizeof(dataBuf.input));
    Serial.write((uint8_t*)dataBuf.angle, sizeof(dataBuf.angle));
    Serial.flush();
    Serial.write(DEVICE_DATA_STREAM_END);
    return RESULT_OK;
}