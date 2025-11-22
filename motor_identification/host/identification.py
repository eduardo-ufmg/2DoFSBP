import sys
import numpy as np
import pandas as pd
import json
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# -------------------- USER PARAMETERS --------------------
J = 0.000153  # [kg m^2]
resample_fs = 100.0  # target Hz for uniform resampling
savgol_window_seconds = 0.02  # smoothing window (sec) for savgol
savgol_poly = 3
na = 4  # ARX: number of output (tau) lags
nb = 4  # ARX: number of input (u) lags (use u[k-1]..u[k-nb])
iter_max = 12
tolerance = 1e-5
# ---------------------------------------------------------

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    t = np.asarray(df.iloc[:,0], dtype=float)
    u = np.asarray(df.iloc[:,1], dtype=float)
    angle = np.asarray(df.iloc[:,2], dtype=float)
    # ensure ascending time
    order = np.argsort(t)
    t, u, angle = t[order], u[order], angle[order]
    # detect units: if max angle > 2*pi assume degrees
    ang_range = np.max(angle) - np.min(angle)
    if ang_range > 2*np.pi + 1e-3 and np.max(angle) > 360 - 1:
        angle = np.deg2rad(angle)
    # unwrap
    angle = np.unwrap(angle)
    return t, u, angle

def resample_to_uniform(t, x, fs):
    t_uniform = np.arange(t[0], t[-1], 1.0/fs)
    f = interp1d(t, x, axis=0, kind='linear')
    return t_uniform, f(t_uniform)

def compute_omega_alpha(angle, fs, win_sec, poly):
    win_pts = int(np.ceil(win_sec * fs))
    if win_pts % 2 == 0: win_pts += 1
    if win_pts < poly+2:
        win_pts = poly+2 + (1 - (poly+2)%2)
    # smooth angle
    ang_smooth = savgol_filter(angle, window_length=win_pts, polyorder=poly)
    # differentiate
    dt = 1.0/fs
    omega = np.gradient(ang_smooth, dt)
    alpha = np.gradient(omega, dt)
    return ang_smooth, omega, alpha

def build_arx_regressors(u, y, na, nb):
    # y: target torque series (length N)
    N = len(y)
    max_lag = max(na, nb)
    rows = N - max_lag
    Phi = np.zeros((rows, na + nb))
    Y = np.zeros(rows)
    for i in range(rows):
        idx = i + max_lag
        # output regressors: -y[k-1], -y[k-2], ...
        for j in range(na):
            Phi[i, j] = -y[idx - (j+1)]
        # input regressors: u[k-1], u[k-2], ...
        for j in range(nb):
            Phi[i, na + j] = u[idx - (j+1)]
        Y[i] = y[idx]
    return Phi, Y

def estimate_arx(u, y, na, nb):
    Phi, Y = build_arx_regressors(u, y, na, nb)
    theta, *_ = lstsq(Phi, Y, rcond=None)
    # split
    a = theta[:na]
    b = theta[na:]
    return a, b

def simulate_arx(u, a, b, y0=None):
    na = len(a)
    nb = len(b)
    N = len(u)
    y = np.zeros(N)
    # initialize with zeros or given y0
    if y0 is not None:
        y[:len(y0)] = y0
    for k in range(max(na, nb), N):
        # -a1*y[k-1] - ... + b1*u[k-1] + ...
        val = 0.0
        for j in range(na):
            val += -a[j] * y[k - (j+1)]
        for j in range(nb):
            val += b[j] * u[k - (j+1)]
        y[k] = val
    return y

def fit_friction(residual, omega):
    # residual ≈ b*omega + tau_c*sign(omega) + tau0
    S = np.vstack([omega, np.sign(omega), np.ones_like(omega)]).T
    theta, *_ = lstsq(S, residual, rcond=None)
    b_est, tau_c_est, tau0_est = theta
    return b_est, tau_c_est, tau0_est

def main(csv_path):
    t, u_in, angle = load_and_prepare(csv_path)
    t_u, u = resample_to_uniform(t, u_in, resample_fs)
    _, angle_r = resample_to_uniform(t, angle, resample_fs)
    ang_smooth, omega, alpha = compute_omega_alpha(angle_r, resample_fs, savgol_window_seconds, savgol_poly)
    # measurable term y_meas = J*alpha
    y_meas = J * alpha
    # iterative estimation
    b = 0.0; tau_c = 0.0; tau0 = 0.0
    prev_err = np.inf
    for it in range(iter_max):
        tau_est = y_meas + b*omega + tau_c*np.sign(omega) + tau0
        # ARX fit u -> tau_est
        a, bb = estimate_arx(u, tau_est, na, nb)
        tau_pred = simulate_arx(u, a, bb)
        # compute residual only for valid indices (same as regression region)
        maxlag = max(na, nb)
        residual = tau_est[maxlag:] - tau_pred[maxlag:]
        omega_reg = omega[maxlag:]
        # fit friction from residual
        b_new, tau_c_new, tau0_new = fit_friction(residual, omega_reg)
        # check convergence (change of parameters)
        param_change = np.abs(b_new - b) + np.abs(tau_c_new - tau_c) + np.abs(tau0_new - tau0)
        b, tau_c, tau0 = b_new, tau_c_new, tau0_new
        # error metric
        err = np.mean(residual**2)
        if np.abs(prev_err - err) < tolerance and param_change < 1e-6:
            break
        prev_err = err
    # final outputs
    print("Estimated friction + offset:")
    print(f"  viscous b = {b:.6g} [Nm/(rad/s)]")
    print(f"  Coulomb tau_c = {tau_c:.6g} [Nm]")
    print(f"  offset tau0 = {tau0:.6g} [Nm]")
    print("ARX parameters:")
    print("  a (output) =", a)
    print("  b (input)  =", bb)
    # Plot diagnostics
    tau_final = y_meas + b*omega + tau_c*np.sign(omega) + tau0
    tau_model = simulate_arx(u, a, bb)
    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.plot(t_u, u, label='u'); plt.legend(); plt.ylabel('u')
    plt.subplot(312)
    plt.plot(t_u, omega, label='omega'); plt.legend(); plt.ylabel('rad/s')
    plt.subplot(313)
    plt.plot(t_u, tau_final, label='tau_est')
    plt.plot(t_u, tau_model, '--', label='tau_ARX_pred')
    plt.legend(); plt.ylabel('Nm'); plt.xlabel('time [s]')
    plt.tight_layout()
    plt.show()
    # save model parameters to files
    np.savez("identified_model.npz", J=J, b=b, tau_c=tau_c, tau0=tau0, a=a, b_arx=bb)
    print("Saved identified_model.npz")
    with open("identified_model.json", "w") as f:
        json.dump({
            "J": J,
            "b": b,
            "tau_c": tau_c,
            "tau0": tau0,
            "a": a.tolist(),
            "b_arx": bb.tolist()
        }, f, indent=4)
    print("Saved identified_model.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python identification.py data.csv")
        sys.exit(1)
    main(sys.argv[1])
