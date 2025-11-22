/**
 * @file MotorFeedforward.cpp
 * @brief Implementation file for the MotorFeedforward class.
 */

#include "MotorFeedForward.h"
#include <math.h> // For isnan

MotorFeedforward::MotorFeedforward(float k, float b, float c, float outputMin, float outputMax)
    : _k(k), _b(b), _c(c), _outputMin(outputMin), _outputMax(outputMax)
{
    // Prevent division by zero if K is invalid
    if (fabs(_k) < 1e-9) {
        _k = 1.0;
    }

    // Ensure output limits are valid
    if (_outputMin > _outputMax) {
        float temp = _outputMin;
        _outputMin = _outputMax;
        _outputMax = temp;
    }
}

float MotorFeedforward::compute(float targetTorque, float currentSpeed)
{
    // Handle invalid inputs
    if (isnan(targetTorque))
        targetTorque = 0.0f;
    if (isnan(currentSpeed))
        currentSpeed = 0.0f;

    // Model Equation: tau_net = K*u - b*omega + c
    // Inverted:       K*u = tau_net + b*omega - c
    //                 u = (tau_net + b*omega - c) / K

    // 1. Calculate the friction/back-EMF force to overcome
    // Note: We use the sign of the speed if we wanted Coulomb friction,
    // but your identified model provided a constant linear offset 'c'.
    float resistiveTorque = (_b * currentSpeed) - _c;

    // 2. Calculate total electrical torque needed
    float totalTorqueRequired = targetTorque + resistiveTorque;

    // 3. Convert torque to input units (u)
    float u = totalTorqueRequired / _k;

    // 4. Clamp output
    return _clamp(u);
}

void MotorFeedforward::setOutputLimits(float outputMin, float outputMax)
{
    if (outputMin > outputMax) {
        return; // Invalid limits
    }
    _outputMin = outputMin;
    _outputMax = outputMax;
}

void MotorFeedforward::setParameters(float k, float b, float c)
{
    // Prevent K=0
    if (fabs(k) < 1e-9) {
        k = 1.0;
    }
    _k = k;
    _b = b;
    _c = c;
}

float MotorFeedforward::_clamp(float value)
{
    if (value > _outputMax)
        return _outputMax;
    if (value < _outputMin)
        return _outputMin;
    return value;
}