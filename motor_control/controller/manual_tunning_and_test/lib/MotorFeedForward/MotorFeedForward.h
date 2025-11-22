/**
 * @file MotorFeedforward.h
 * @brief Arduino library for model-based feedforward motor control.
 * @author Eduardo Henrique Basilio de Carvalho
 * @date 2025/11/22
 *
 * This library provides a feedforward controller that inverts the identified
 * DC/BLDC motor dynamics. It calculates the necessary control input (u) to
 * achieve a desired mechanical torque given the current speed, canceling out
 * friction and back-EMF.
 */

#ifndef MOTORFEEDFORWARD_H
#define MOTORFEEDFORWARD_H

#include <Arduino.h>

/**
 * @class MotorFeedforward
 * @brief Feedforward controller based on identified system dynamics.
 *
 * Implements the inversion of the dynamic model:
 * J*alpha = (K * u) - (b * omega) + c
 *
 * To achieve a target net torque (tau_ref = J*alpha), the required input is:
 * u = (tau_ref + b * omega - c) / K
 */
class MotorFeedforward
{
public:
    /**
     * @brief Constructor for the Feedforward controller.
     *
     * @param k Input gain constant [Nm / unit_input].
     * @param b Viscous friction/damping coefficient [Nm / (rad/s)].
     * @param c Constant bias/offset torque [Nm].
     * @param outputMin Minimum output limit (default: -1.0).
     * @param outputMax Maximum output limit (default: 1.0).
     */
    MotorFeedforward(float k, float b, float c, float outputMin = -1.0, float outputMax = 1.0);

    /**
     * @brief Computes the required control signal.
     *
     * Calculates the 'u' required to generate the 'targetTorque' while
     * overcoming the friction present at 'currentSpeed'.
     *
     * @param targetTorque The desired net mechanical torque [Nm].
     * @param currentSpeed The current motor speed [rad/s].
     * @return The computed feedforward control output (clamped to limits).
     */
    float compute(float targetTorque, float currentSpeed);

    /**
     * @brief Sets the output limits.
     *
     * @param outputMin Minimum output limit.
     * @param outputMax Maximum output limit.
     */
    void setOutputLimits(float outputMin, float outputMax);

    /**
     * @brief Updates the physical model parameters.
     *
     * @param k Input gain constant.
     * @param b Damping coefficient.
     * @param c Constant bias.
     */
    void setParameters(float k, float b, float c);

private:
    // Model parameters
    float _k; ///< Input Gain (K)
    float _b; ///< Damping/Friction (b)
    float _c; ///< Constant Bias (c)

    // Output limits
    float _outputMin;
    float _outputMax;

    /**
     * @brief Clamps a value to the output limits.
     *
     * @param value The value to clamp.
     * @return The clamped value.
     */
    float _clamp(float value);
};

#endif // MOTORFEEDFORWARD_H