#include <AntiWindupPID.h>
#include <Arduino.h>
#include <Comms.h>
#include <Nidec24H.h>

const unsigned int testDataLength = 1024;
const unsigned int samplePeriodMs = 10;
const unsigned int referenceChangeTimeMs = 200;

Nidec24H motor(27, 26, 25, 33, 32, 20000, 8, 100, 0.000153);

ResultCode runMotorTest();
ResultCode sendTestData();

typedef struct
{
    float reference[testDataLength];
    float controlEffort[testDataLength];
    float torqueEstimate[testDataLength];
} TestData;

TestData testData;
ResultCode testResult = RESULT_ERROR;

AntiWindupPID pidController(5.0f, 900.0f, 0.0f, -1.0f, 1.0f);

void setup()
{

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
    motor.brake(false);
    motor.setSpeed(0.0f);

    delay(1000); // Allow some time for motor to start

    unsigned int testStartTimeMs = millis();
    unsigned int referenceChangeLastTimeMs = testStartTimeMs,
                 referenceChangeCurrentTimeMs = testStartTimeMs;
    unsigned int sampleLastTimeMs = testStartTimeMs, sampleCurrentTimeMs = testStartTimeMs;

    motor.resetEncoder();
    pidController.setMode(true);

    float reference = 0.0f;
    float controlEffort = 0.0f;
    float torqueEstimate = 0.0f;

    float aux_angle = 0.0f;

    for (unsigned int i = 0; i < testDataLength; i++) {

        // Update reference every referenceChangeTimeMs milliseconds
        referenceChangeCurrentTimeMs = millis();
        if (referenceChangeCurrentTimeMs - referenceChangeLastTimeMs >= referenceChangeTimeMs) {
            reference = sin(i) / 100.0f; // Reference between -0.01 and +0.01 Nm
            referenceChangeLastTimeMs = referenceChangeCurrentTimeMs;
        }

        aux_angle = motor.readAngle();

        torqueEstimate = motor.estimateTorque();

        controlEffort = pidController.compute(reference, torqueEstimate);

        motor.setSpeed(controlEffort);

        testData.reference[i] = reference;
        testData.controlEffort[i] = controlEffort;
        testData.torqueEstimate[i] = torqueEstimate;

        do {
            sampleCurrentTimeMs = millis();
            yield(); // To keep watchdog happy
        } while (sampleCurrentTimeMs - sampleLastTimeMs < samplePeriodMs);
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

    Serial.write((uint8_t *)testData.reference, sizeof(testData.reference));
    Serial.write((uint8_t *)testData.controlEffort, sizeof(testData.controlEffort));
    Serial.write((uint8_t *)testData.torqueEstimate, sizeof(testData.torqueEstimate));

    Serial.flush();

    Serial.write(DEVICE_DATA_STREAM_END);
    return RESULT_OK;
}
