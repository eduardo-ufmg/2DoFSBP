#ifndef COMMS_H
#define COMMS_H

#include <Arduino.h>

typedef enum {
    HOST_CHECK_CONNECTION = 0x01,
    DEVICE_CHECK_CONNECTION = 0x02,
    HOST_START_TEST = 0x03,
    DEVICE_ACK_START = 0x04,
    DEVICE_TEST_SUCCESS = 0x05,
    HOST_REQUEST_DATA = 0x06,
    DEVICE_DATA_REQUEST_ACK = 0x07,
} CommCode;

static const char DEVICE_DATA_STREAM_START[] = "DATA_START";
static const char DEVICE_DATA_STREAM_END[] = "DATA_END";

typedef enum {
    RESULT_OK = 0x00,
    RESULT_ERROR = 0x01,
} ResultCode;

ResultCode waitForConnectionCheck();
ResultCode answerConnectionCheck();
ResultCode connectionCheck();
ResultCode waitForStartCommand();
ResultCode ackStartCommand();

ResultCode sendSuccessMessage();
ResultCode waitForDataRequest();
ResultCode ackDataRequest();

#endif // COMMS_H
