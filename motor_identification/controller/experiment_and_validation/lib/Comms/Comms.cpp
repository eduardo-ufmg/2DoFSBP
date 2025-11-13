#include <Comms.h>

ResultCode waitForConnectionCheck()
{
    while (true) {
        if (Serial.available() > 0) {
            uint8_t code = Serial.read();
            if (code == HOST_CHECK_CONNECTION) {
                return RESULT_OK;
            }
        }
        yield();
    }
}

ResultCode answerConnectionCheck()
{
    Serial.write(DEVICE_CHECK_CONNECTION);
    return RESULT_OK;
}

ResultCode connectionCheck()
{
    if (waitForConnectionCheck() != RESULT_OK) {
        return RESULT_ERROR;
    }
    if (answerConnectionCheck() != RESULT_OK) {
        return RESULT_ERROR;
    }
    return RESULT_OK;
}

ResultCode waitForStartCommand()
{
    while (true) {
        if (Serial.available() > 0) {
            uint8_t code = Serial.read();
            if (code == HOST_START_TEST) {
                return RESULT_OK;
            }
        }
        yield();
    }

    return RESULT_ERROR;
}

ResultCode ackStartCommand()
{
    Serial.write(DEVICE_ACK_START);
    return RESULT_OK;
}

ResultCode sendSuccessMessage()
{
    Serial.write(DEVICE_TEST_SUCCESS);
    return RESULT_OK;
}

ResultCode waitForDataRequest()
{
    while (true) {
        if (Serial.available() > 0) {
            uint8_t code = Serial.read();
            if (code == HOST_REQUEST_DATA) {
                return RESULT_OK;
            }
        }
        yield();
    }

    return RESULT_ERROR;
}

ResultCode ackDataRequest()
{
    Serial.write(DEVICE_DATA_REQUEST_ACK);
    return RESULT_OK;
}
