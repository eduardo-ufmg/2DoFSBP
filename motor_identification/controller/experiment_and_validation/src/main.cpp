#include <Arduino.h>
#include <Nidec24H.h>
#include <Comms.h>

const unsigned int testDataLength = 4096;
const unsigned int samplePeriodMs = 10;
const unsigned int inputChangeTimeMs = 200;

Nidec24H motor(27, 26, 25, 33, 32, 20000, 8, 100);

ResultCode runMotorTest();
ResultCode sendTestData();

typedef struct {
    float input[testDataLength];
    float angle[testDataLength];
} TestData;

TestData testData;
ResultCode testResult = RESULT_ERROR;

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

void loop() {

    int ledHalfPeriodMs = (testResult == RESULT_OK) ? 200 : 1000;

    digitalWrite(LED_BUILTIN, HIGH);
    delay(ledHalfPeriodMs);
    digitalWrite(LED_BUILTIN, LOW);
    delay(ledHalfPeriodMs);
}

ResultCode runMotorTest()
{

    unsigned int lastTimeMs = millis(), currentTimeMs = millis();

    float inputValue = 0.0f;

    motor.brake(false);
    motor.setSpeed(inputValue);

    for (unsigned int i = 0; i < testDataLength; i++) {
        testData.input[i] = inputValue;
        testData.angle[i] = motor.readAngle();

        currentTimeMs = millis();
        if (currentTimeMs - lastTimeMs >= inputChangeTimeMs) {
            inputValue = (static_cast<float>(esp_random()) / UINT32_MAX) / 2.0f - 0.25f; // Random value between -0.25 and +0.25
            motor.setSpeed(inputValue);
            lastTimeMs = currentTimeMs;
        }

        delay(samplePeriodMs);
    }

    motor.setSpeed(0.0f);
    motor.brake(true);

    return RESULT_OK;
}

ResultCode sendTestData()
{
    Serial.write(DEVICE_DATA_STREAM_START);

    Serial.flush();

    Serial.write((uint8_t*)testData.input, sizeof(testData.input));
    Serial.write((uint8_t*)testData.angle, sizeof(testData.angle));

    Serial.flush();

    Serial.write(DEVICE_DATA_STREAM_END);
    return RESULT_OK;
}