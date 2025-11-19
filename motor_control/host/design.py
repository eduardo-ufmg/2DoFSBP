"""
Reads an identified model JSON,
designs a cascaded inner torque controller (torque-setpoint -> motor duty u in [-1,1]),
and simulates the closed-loop torque tracking using the identified continuous-time
plant:
    omega_dot = a*u + b*omega + c*sign(omega) + d
Torque estimate used for feedback:
    tau_est = I_w * (a*u + b*omega + c*sign(omega) + d)

Controller structure (per-sample discrete algebraic implementation):
    - Feedforward: u_ff = tau_ref / K_tau   (K_tau = I_w * a)
    - PI feedback on torque error, implemented algebraically to avoid implicit loops:
      Let tau0 = I_w*(b*omega + c*sign(omega) + d)
      Then tau_est = K_tau * u + tau0  => linear in u
      The controller computes:
        u = (u_ff + Kp*(tau_ref - tau_est_prev) + Ki*I_int) / (1 + Kp*K_tau)
      where I_int is discrete integral of torque error.
    - The implementation uses tau_est_prev when computing u; because tau_est depends linearly on u
      we rearrange to the explicit formula above (tau_est on RHS replaced by K_tau*u + tau0).
      This yields the explicit algebraic update used below.

Usage:
    python design.py \
        --json identified_model.json \
        [--fs 100] \
        [--f_bw 10.0] \
        [--sim_time 1.0] \
        [--tau_step 0.05] \
        [--out_prefix inner_loop_]

Outputs:
    - Prints designed gains and key numbers.
    - Saves plots: <out_prefix>torque_tracking.png, <out_prefix>control_signals.png
    - Saves a small JSON summary <out_prefix>summary.json

Requires: numpy, pandas, matplotlib, json, argparse
"""

import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_identified_model(path):
    with open(path, 'r') as f:
        j = json.load(f)
    coeffs = j.get('coefficients', {})
    inertia_info = j.get('inertia_results', None)
    return j, coeffs, inertia_info

def sign_with_deadzone(x, dz=1e-6):
    s = np.sign(x)
    s[np.abs(x) <= dz] = 0.0
    return s

def design_pi_gains(K_tau, f_bw_hz, f_int_frac=10.0):
    """
    Simple heuristic design for PI on torque loop using static gain K_tau (N·m per unit input).
    - target closed-loop bandwidth f_bw_hz (Hz)
    - integral corner placed f_bw_hz / f_int_frac
    Returns (Kp, Ki) continuous-time gains where Ki has units N·m/s per unit-error? 
    In this implementation Ki is the discrete integrator coefficient scaled for dt in sim.
    """
    if K_tau <= 0:
        raise ValueError("K_tau must be positive for design.")
    wb = 2.0 * math.pi * f_bw_hz
    # Kp chosen so closed-loop time-constant ~ 1/wb: Kp * K_tau ≈ wb  => Kp = wb / K_tau
    Kp = wb / K_tau
    # integrator frequency (rad/s)
    wi = wb / f_int_frac
    # Ki (continuous) = Kp * wi
    Ki = Kp * wi
    return float(Kp), float(Ki)

def simulate_inner_loop(a, b, c, d, I_w, K_tau, Kp, Ki, fs, sim_time, tau_ref_waveform, u_sat=1.0):
    dt = 1.0 / fs
    steps = int(sim_time * fs)
    t = np.arange(steps) * dt

    omega = 0.0
    theta = 0.0
    u = 0.0
    I_int = 0.0

    rec = {
        't': t,
        'tau_ref': np.zeros(steps),
        'tau_meas': np.zeros(steps),
        'u': np.zeros(steps),
        'omega': np.zeros(steps),
        'theta': np.zeros(steps),
        'tau0': np.zeros(steps)
    }

    for k in range(steps):
        tau_ref = tau_ref_waveform(k*dt)
        # feedforward (algebraic inverse of nominal static gain)
        if K_tau == 0:
            u_ff = 0.0
        else:
            u_ff = tau_ref / K_tau

        # compute tau0 = I_w*(b*omega + c*sign(omega) + d)
        sgn = 0.0 if abs(omega) < 1e-8 else math.copysign(1.0, omega)
        tau0 = I_w * (b * omega + c * sgn + d)

        # torque estimate is tau_est = K_tau * u + tau0
        # The controller algebraic explicit formula:
        # u = (u_ff + Kp*(tau_ref - tau0) + Ki*I_int) / (1 + Kp*K_tau)
        numerator = u_ff + Kp * (tau_ref - tau0) + Ki * I_int
        denom = 1.0 + Kp * K_tau
        u_new = numerator / denom

        # saturate u
        if u_new > u_sat:
            u_new = u_sat
        elif u_new < -u_sat:
            u_new = -u_sat

        # now compute plant update
        # omega_dot = a*u + b*omega + c*sign(omega) + d
        sgn_omega = 0.0 if abs(omega) < 1e-8 else math.copysign(1.0, omega)
        omega_dot = a * u_new + b * omega + c * sgn_omega + d
        omega = omega + omega_dot * dt
        theta = theta + omega * dt

        # compute measured torque (as used for feedback next step)
        tau_meas = I_w * (a * u_new + b * omega + c * sgn_omega + d)

        # update integral with current torque error (forward Euler)
        err = tau_ref - tau_meas
        I_int = I_int + err * dt

        # record
        rec['tau_ref'][k] = tau_ref
        rec['tau_meas'][k] = tau_meas
        rec['u'][k] = u_new
        rec['omega'][k] = omega
        rec['theta'][k] = theta
        rec['tau0'][k] = tau0

        u = u_new

    return rec

def default_step_waveform(tau_step, t_step):
    def wf(t):
        return tau_step if t >= t_step else 0.0
    return wf

def main():
    parser = argparse.ArgumentParser(description="Design inner torque loop from identified model JSON.")
    parser.add_argument('--json', default='identified_model.json', help='Path to identified_model.json')
    parser.add_argument('--fs', type=float, default=None, help='Sampling frequency for simulation (Hz). If omitted, uses JSON sampling_frequency_Hz or 100 Hz.')
    parser.add_argument('--f_bw', type=float, default=10.0, help='Desired torque-loop bandwidth (Hz). Default 10 Hz.')
    parser.add_argument('--sim_time', type=float, default=1.0, help='Simulation time (s).')
    parser.add_argument('--tau_step', type=float, default=0.05, help='Torque step magnitude for simulation (N·m).')
    parser.add_argument('--tau_step_time', type=float, default=0.05, help='Time at which torque step is applied (s).')
    parser.add_argument('--out_prefix', default='inner_loop_', help='Prefix for output files.')
    parser.add_argument('--u_sat', type=float, default=1.0, help='Motor command saturation (default 1.0).')
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        print(f"ERROR: JSON file not found: {args.json}", file=sys.stderr)
        sys.exit(2)

    j, coeffs, inertia = load_identified_model(args.json)

    a = float(coeffs.get('u', coeffs.get('a', coeffs.get('U', 0.0))))
    b = float(coeffs.get('omega', coeffs.get('b', 0.0)))
    c = float(coeffs.get('sign_omega', coeffs.get('c', 0.0)))
    d = float(coeffs.get('const', coeffs.get('d', 0.0)))

    if inertia is None:
        print("ERROR: identified JSON must include 'inertia_results' with I_w and K_tau_per_input.", file=sys.stderr)
        sys.exit(2)

    I_w = float(inertia.get('I_w', None))
    K_tau = float(inertia.get('K_tau_per_input', None))

    if I_w is None or K_tau is None:
        print("ERROR: inertia_results missing I_w or K_tau_per_input.", file=sys.stderr)
        sys.exit(2)

    fs = args.fs if args.fs is not None else float(j.get('sampling_frequency_Hz', 100.0))
    f_bw = float(args.f_bw)

    # basic sanity checks
    if K_tau == 0.0:
        print("ERROR: K_tau is zero; cannot design feedforward/inversion.", file=sys.stderr)
        sys.exit(2)
    if f_bw <= 0:
        print("ERROR: f_bw must be positive.", file=sys.stderr)
        sys.exit(2)

    # design PI gains (continuous-time heuristic)
    Kp_cont, Ki_cont = design_pi_gains(K_tau, f_bw_hz=f_bw, f_int_frac=10.0)
    # For discrete-time integrator in simulation we use Ki_cont as continuous Ki (rad/s), and convert during update by multiplying I_int*dt earlier (we already multiplied error * dt)
    # In our algebraic controller expression we used Ki*I_int where I_int is integral of error (N·m·s). So Ki_cont is used directly.
    Kp = Kp_cont
    Ki = Ki_cont

    # simulate inner loop responding to torque step
    tau_step = float(args.tau_step)
    sim_time = float(args.sim_time)
    waveform = default_step_waveform(tau_step, args.tau_step_time)

    rec = simulate_inner_loop(a, b, c, d, I_w, K_tau, Kp, Ki, fs, sim_time, waveform, u_sat=args.u_sat)

    # compute simple metrics: rise time (10-90), steady-state error
    tau_ref = rec['tau_ref']
    tau_meas = rec['tau_meas']
    t = rec['t']
    # find indices after step time
    step_idx = np.where(t >= args.tau_step_time)[0][0]
    tau_target = tau_step
    # steady-state window: last 20% of sim
    ss_start = int(len(t)*0.8)
    ss_error = tau_target - np.mean(tau_meas[ss_start:])
    # rise time 10-90%
    try:
        idx10 = step_idx + np.where(tau_meas[step_idx:] >= 0.1 * tau_target)[0][0]
        idx90 = step_idx + np.where(tau_meas[step_idx:] >= 0.9 * tau_target)[0][0]
        rise_time = t[idx90] - t[idx10]
    except Exception:
        rise_time = None

    # save plots
    prefix = args.out_prefix
    plt.figure(figsize=(8,4))
    plt.plot(t, tau_ref, label='tau_ref (N·m)')
    plt.plot(t, tau_meas, label='tau_meas (N·m)')
    plt.xlabel('time (s)')
    plt.ylabel('Torque (N·m)')
    plt.title('Torque tracking')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prefix + 'torque_tracking.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(t, rec['u'], label='u (command)')
    plt.plot(t, rec['omega'], label='omega (rad/s)')
    plt.xlabel('time (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prefix + 'control_signals.png', dpi=200)
    plt.close()

    # save summary
    summary = {
        'identified_json_used': os.path.abspath(args.json),
        'model_coefficients': {'a': a, 'b': b, 'c': c, 'd': d},
        'I_w': I_w,
        'K_tau_per_input': K_tau,
        'design_parameters': {'f_bw_Hz': f_bw, 'Kp': Kp, 'Ki': Ki},
        'simulation': {
            'sim_time_s': sim_time,
            'tau_step_Nm': tau_step,
            'tau_step_time_s': args.tau_step_time,
            'steady_state_error_Nm': float(ss_error),
            'rise_time_s_10_90': float(rise_time) if rise_time is not None else None
        },
        'notes': (
            "Controller uses algebraic explicit implementation that accounts for the "
            "identified static gain K_tau and tau0 (omega-dependent term). "
            "Tune f_bw and integrator fraction if needed."
        )
    }
    with open(prefix + 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # print concise results
    print("INNER TORQUE LOOP DESIGN SUMMARY")
    print(f"  JSON used: {os.path.abspath(args.json)}")
    print(f"  model: omega_dot = a*u + b*omega + c*sign(omega) + d")
    print(f"    a = {a:.6g}, b = {b:.6g}, c = {c:.6g}, d = {d:.6g}")
    print(f"  inertia I_w = {I_w:.6g} kg·m^2, K_tau_per_input = {K_tau:.6g} N·m per unit input")
    print(f"  desired torque-loop bandwidth = {f_bw:.3g} Hz")
    print(f"  designed PI gains: Kp = {Kp:.6g}, Ki = {Ki:.6g} (continuous-time heuristic)")
    print(f"  simulation: torque step {tau_step} N·m at t={args.tau_step_time}s, sim_time={sim_time}s")
    if rise_time is not None:
        print(f"  measured rise time (10->90%) = {rise_time:.4f} s")
    else:
        print("  rise time: could not compute (insufficient excursion or saturation)")
    print(f"  steady-state torque error (last 20% window) = {ss_error:.6g} N·m")
    print(f"  output files: {prefix}torque_tracking.png, {prefix}control_signals.png, {prefix}summary.json")

if __name__ == '__main__':
    main()
