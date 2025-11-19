"""
Reads `experiment_data.csv` (columns: "Input", "Angle" in radians; sampled at 100 Hz),
estimates the flywheel/motor dynamics from input (u in [-1,1]) to motor-axis angle (encoder),
and saves an identified linear model.

Model used (continuous-time, lumped):
    ω = d/dt(angle)
    ω_dot = a*u + b*ω + c*sign(ω) + d

Interpretation:
    a  : angular-acceleration gain per unit input (rad/s^2 per unit u)
    b  : viscous coefficient (rad/s per s) (expected negative)
    c  : coulomb-like friction torque term (entered as acceleration term)
    d  : acceleration bias (e.g., constant torque / disturbance per I_w)

If you provide the flywheel geometry (mass, inner_radius, outer_radius), the script will
convert the fitted acceleration/gain parameters into torque units using:
    I_w = 0.5 * m * (r_o^2 + r_i^2)   (moment of inertia of a thin homogeneous ring)
and will report:
    K_tau = a * I_w   (estimated torque-per-input-unit, in N·m per unit input)
    B_viscous = -b * I_w   (viscous damping torque coefficient, N·m·s/rad)
    Tau_coulomb = c * I_w * sign(ω) (estimate magnitude)

Outputs:
  - plots saved as PNG (input, angle, velocity, fit vs measured, residuals)
  - JSON file `identified_model.json` with model parameters and simple metrics
  - console printout with summary

Dependencies: numpy, pandas, scipy, matplotlib, sklearn
Usage:
    python identify_flywheel_dynamics.py \
        --csv experiment_data.csv \
        [--fs 100] \
        [--mass 0.5 --r_in 0.02 --r_out 0.06] \
        [--window_sec 0.1] \
        [--out identified_model.json]
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import sys

def nearest_odd(n):
    n = int(n)
    if n % 2 == 0:
        n += 1
    return max(3, n)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns "Input" and "Angle"
    if 'Input' not in df.columns or 'Angle' not in df.columns:
        raise ValueError("CSV must contain headers 'Input' and 'Angle'.")
    u = df['Input'].to_numpy(dtype=float)
    angle = df['Angle'].to_numpy(dtype=float)
    return u, angle

def smooth_and_derivatives(angle, fs, window_sec=0.1, polyorder=3):
    # choose window length in samples (odd)
    wl = nearest_odd(window_sec * fs)
    # ensure wl < len(angle)
    if wl >= len(angle):
        wl = nearest_odd(max(3, len(angle)//2))
    # smoothed angle
    angle_s = savgol_filter(angle, wl, polyorder, mode='interp')
    # first derivative: angular velocity (rad/s)
    omega = savgol_filter(angle, wl, polyorder, deriv=1, delta=1.0/fs, mode='interp')
    # second derivative: angular acceleration (rad/s^2)
    omega_dot = savgol_filter(angle, wl, polyorder, deriv=2, delta=1.0/fs, mode='interp')
    return angle_s, omega, omega_dot, wl

def build_regressors(u, omega, include_coulomb=True):
    # regressors: [u, omega, sign(omega), 1]
    sign_omega = np.sign(omega)
    # treat small velocities as zero to avoid noisy sign jitter
    small_vel_mask = np.abs(omega) < 1e-4
    sign_omega[small_vel_mask] = 0.0
    if include_coulomb:
        X = np.column_stack([u, omega, sign_omega, np.ones_like(u)])
        reg_names = ['u', 'omega', 'sign_omega', 'const']
    else:
        X = np.column_stack([u, omega, np.ones_like(u)])
        reg_names = ['u', 'omega', 'const']
    return X, reg_names

def fit_linear_model(X, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model.coef_, y_pred, r2

def compute_inertia_ring(m, r_in, r_out):
    # moment of inertia about center axis for homogeneous ring (shell)
    # More generally for ring with finite thickness: I = 0.5 * m * (r_out^2 + r_in^2)
    return 0.5 * m * (r_out**2 + r_in**2)

def save_json(outpath, data):
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Identify flywheel/motor dynamics from input-angle data.")
    parser.add_argument('--csv', required=True, help='Path to experiment_data.csv')
    parser.add_argument('--fs', type=float, default=100.0, help='Sampling frequency in Hz (default: 100)')
    parser.add_argument('--mass', type=float, default=None, help='Flywheel mass in kg (optional)')
    parser.add_argument('--r_in', type=float, default=None, help='Flywheel inner radius in meters (optional)')
    parser.add_argument('--r_out', type=float, default=None, help='Flywheel outer radius in meters (optional)')
    parser.add_argument('--window_sec', type=float, default=0.12, help='Savitzky-Golay window length in seconds (default 0.12s)')
    parser.add_argument('--out', default='identified_model.json', help='Output JSON file for identified parameters')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    u, angle = load_data(args.csv)
    n = len(u)
    t = np.arange(n) / args.fs

    # smoothing + derivatives
    angle_s, omega, omega_dot, wl = smooth_and_derivatives(angle, args.fs, window_sec=args.window_sec)
    print(f"Data loaded: {n} samples, fs={args.fs} Hz, SG window={wl} samples")

    # build regressors for omega_dot = a*u + b*omega + c*sign(omega) + d
    X, reg_names = build_regressors(u, omega, include_coulomb=True)
    y = omega_dot

    # Fit coefficients (least squares linear regression)
    coef, y_pred, r2 = fit_linear_model(X, y)
    coef_dict = {name: float(val) for name, val in zip(reg_names, coef)}

    # if user provided inertia info, compute torque-related params
    inertia_info = None
    if args.mass is not None and args.r_in is not None and args.r_out is not None:
        I_w = compute_inertia_ring(args.mass, args.r_in, args.r_out)
        # coefficients map: omega_dot = a*u + b*omega + c*sign + d
        a = coef_dict['u']
        b = coef_dict['omega']
        c = coef_dict['sign_omega']
        d = coef_dict['const']
        # torque = I_w * alpha, so torque per unit input:
        K_tau = a * I_w
        B_viscous = -b * I_w   # viscous damping torque (positive if b<0)
        Tau_coulomb = c * I_w  # signed coulomb torque acceleration -> torque
        Tau_bias = d * I_w
        inertia_info = {
            'I_w': float(I_w),
            'K_tau_per_input': float(K_tau),
            'B_viscous_Nm_per_rad_per_s': float(B_viscous),
            'Tau_coulomb_Nm': float(Tau_coulomb),
            'Tau_bias_Nm': float(Tau_bias)
        }
    else:
        a = coef_dict.get('u', None)
        b = coef_dict.get('omega', None)
        c = coef_dict.get('sign_omega', None)
        d = coef_dict.get('const', None)

    # Save results to JSON
    output = {
        'sampling_frequency_Hz': args.fs,
        'n_samples': n,
        'sg_window_samples': wl,
        'regressors': reg_names,
        'coefficients': coef_dict,
        'r2_omega_dot_fit': float(r2),
        'inertia_info_provided': args.mass is not None and args.r_in is not None and args.r_out is not None,
        'inertia_results': inertia_info
    }
    save_json(args.out, output)
    print(f"Identification results saved to {args.out}")
    print("Summary:")
    print(f"  fit R^2 (omega_dot): {r2:.4f}")
    print("  coefficients (omega_dot = a*u + b*omega + c*sign(omega) + d):")
    for k,v in coef_dict.items():
        print(f"    {k}: {v:.6g}")
    if inertia_info:
        print("  inferred torque-domain parameters (using provided mass & radii):")
        print(f"    I_w = {inertia_info['I_w']:.6g} kg·m²")
        print(f"    K_tau (N·m per unit input) = {inertia_info['K_tau_per_input']:.6g}")
        print(f"    B_viscous (N·m·s/rad) ≈ {inertia_info['B_viscous_Nm_per_rad_per_s']:.6g}")
        print(f"    Tau_coulomb (N·m) ≈ {inertia_info['Tau_coulomb_Nm']:.6g}")
        print(f"    Tau_bias (N·m) ≈ {inertia_info['Tau_bias_Nm']:.6g}")

    # Plotting: input, angle, velocity, fit vs measured acceleration
    try:
        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(t, u, label='Input (u)')
        plt.ylabel('u')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(t, angle, label='Angle (raw)')
        plt.plot(t, angle_s, label='Angle (smoothed)', linewidth=1)
        plt.ylabel('angle (rad)')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(t, omega, label='omega (rad/s)')
        plt.ylabel('omega')
        plt.xlabel('time (s)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('input_angle_velocity.png', dpi=150)
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(t, y, label='omega_dot (measured)')
        plt.plot(t, y_pred, label='omega_dot (predicted)', alpha=0.8)
        plt.xlabel('time (s)')
        plt.ylabel('alpha (rad/s^2)')
        plt.legend()
        plt.title(f'Fit R^2 = {r2:.4f}')
        plt.tight_layout()
        plt.savefig('accel_fit.png', dpi=150)
        plt.close()

        # residuals
        plt.figure(figsize=(8,3))
        plt.plot(t, y - y_pred)
        plt.xlabel('time (s)')
        plt.ylabel('residual (rad/s^2)')
        plt.title('Residuals: measured - predicted')
        plt.tight_layout()
        plt.savefig('residuals.png', dpi=150)
        plt.close()
        print("Plots saved: input_angle_velocity.png, accel_fit.png, residuals.png")
    except Exception as e:
        print(f"Warning: plotting failed: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
