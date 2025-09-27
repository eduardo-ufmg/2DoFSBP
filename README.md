# 2DoFSBP
A two degrees of freedom self balancing pendulum.
3D Drawing: https://cad.onshape.com/documents/dcaab3bd08b5da9e1fdd55d0/

# TODO

## Printing
- [ ] Print a half-wheel pair
- [ ] Conclude motor identification before proceeding
- [ ] Print and test a motor connector
- [ ] Print and test the base
- [ ] Print and test a connector for the wheel guard
- [ ] Print and test a wheel guard
- [ ] Print and test the second set

## Motor Identification
- [ ] Test motor
- [ ] Gather the following information
  - Is the PWM signal active low or active high?
  - Which level in the motor's DIR pin makes it rotate counterclockwise (from the motor's perspective, rather than facing its front)?
  - When the motor is rotating counterclockwise, does the encoder output from [this api](https://github.com/madhephaestus/ESP32Encoder) increment or decrement? (use attachFullQuad and consider 400 pulses per revolution)
- [ ] Create a class for the motor
  - Input signal must be a float in [-1, 1]
  - -1 → max speed clockwise, 1 → max speed counterclockwise
- [ ] Create a class for the encoder
  - Counterclockwise movement increments its position, clockwise decrements it
- [ ] Test both classes
- [ ] Write the code for the identification procedure
  - [ ] Microcontroller
    - Not supposed to compute speed online. Simply collect and transmit the encoder position readings.
    - Use a series of steps large enough for it to approach steady state, but not enough for it to reach it. (See PRBS identification)
    - Collect the input signal to the motor's object and the output signal from the encoder's object at 10 ms
    - Send data to the host at each sampling
    - Avoid blocking operations
  - [ ] Host
    - Listen for the experiment data
    - Store the experiment data when it is finished
    - Preprocess it
      - Remove invalid lines
      - Compute speed and acceleration from the position
    - Fit a standard DC motor transfer function
    - Store it and any available statistic about it
    - Extend it so U(s) is the input signal to the motor's object and Y(s) is the wheel torque
- [ ] Write the code for the model validation procedure
  - [ ] Microcontroller
    - Use the same code that was used for the identification
  - [ ] Host
    - Same acquisition and preprocessing as in the identification procedure
    - Simulate the model for the collected input
    - Compare its output to the captured output from the system

## Pendulum Modeling
- [ ] Capture a simplified geometric model
- [ ] Model the MIMO system with inputs T1 and T2 and outputs Theta and Phi, where:
  - T1 is the torque of the motor that controls Theta
  - T2 is the torque of the motor that controls Phi
  - Theta is the pitch angle
  - Phi is the roll angle
  - Coupling is to be considered

