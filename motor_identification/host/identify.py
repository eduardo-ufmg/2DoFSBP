from __future__ import annotations
import argparse
import os
import sys
import math
import warnings

from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import control

# -----------------------
# Utilities / I/O
# -----------------------
def read_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    df = pd.read_csv(filename)
    if 'Input' not in df.columns or 'Angle' not in df.columns:
        raise ValueError("CSV must contain columns named 'Input' and 'Angle'.")
    u = df['Input'].astype(float).to_numpy()
    y = df['Angle'].astype(float).to_numpy()
    if u.shape[0] != y.shape[0]:
        raise ValueError("Input and Angle must have the same length.")
    # drop NaNs
    mask = ~(np.isnan(u) | np.isnan(y))
    if not mask.all():
        u = u[mask]
        y = y[mask]
    if len(u) < 10:
        raise ValueError("Not enough valid samples (need >=10).")
    return u, y

def detrend_remove_mean(u: np.ndarray, y: np.ndarray, detrend: bool) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Optionally remove linear trend and/or mean from both signals.
    Returns processed signals and dictionary with removed trends/means for later restoration.
    """
    info = {}
    if detrend:
        # remove linear trend using least squares (y = a*t + b)
        t = np.arange(len(u))
        A = np.vstack([t, np.ones(len(t))]).T
        # input trend
        coeff_u, *_ = np.linalg.lstsq(A, u, rcond=None)
        trend_u = A @ coeff_u
        u = u - trend_u
        info['u_trend_coeff'] = coeff_u.tolist()
        # output trend
        coeff_y, *_ = np.linalg.lstsq(A, y, rcond=None)
        trend_y = A @ coeff_y
        y = y - trend_y
        info['y_trend_coeff'] = coeff_y.tolist()
    # remove mean
    mu_u = float(np.mean(u))
    mu_y = float(np.mean(y))
    u = u - mu_u
    y = y - mu_y
    info['u_mean'] = mu_u
    info['y_mean'] = mu_y
    return u, y, info

# -----------------------
# ARX feature matrix builder
# -----------------------
def build_arx_phi(u: np.ndarray, y: np.ndarray, na: int, nb: int, nk: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build regression matrix Phi and target vector Y for ARX:
        y[k] = a1*y[k-1] + ... + ana*y[k-na] + b1*u[k-nk] + ... + b_nb*u[k-nk-nb+1] + e[k]
    nk is input delay (default 0 -> uses u[k]).
    Returns: Phi (M x (na+nb)), Y (M,), k_start (first sample index used)
    """
    N = len(y)
    # Effective delay: we need indices k - i >= 0 and k - nk - (nb-1) >= 0
    k_start = max(na, nk + nb - 1)
    rows = []
    targets = []
    for k in range(k_start, N):
        phi_row = []
        # past outputs y[k-1]..y[k-na]
        for i in range(1, na + 1):
            phi_row.append(y[k - i])
        # inputs b1..b_nb correspond to u[k-nk], u[k-nk-1], ...
        for j in range(0, nb):
            idx = k - nk - j
            phi_row.append(u[idx] if idx >= 0 else 0.0)
        rows.append(phi_row)
        targets.append(y[k])
    Phi = np.asarray(rows, dtype=float)
    Y = np.asarray(targets, dtype=float)
    return Phi, Y, k_start

# -----------------------
# ARX estimators
# -----------------------
def fit_arx_ols(u: np.ndarray, y: np.ndarray, na: int, nb: int, nk: int = 0) -> dict[str, Any]:
    Phi, Y, k0 = build_arx_phi(u, y, na, nb, nk)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    a = theta[:na].copy()
    b = theta[na:].copy()
    y_pred = Phi @ theta
    res = Y - y_pred
    sigma2 = float(np.var(res, ddof=max(1, Phi.shape[1])))
    return dict(a=a, b=b, theta=theta, k0=k0, res=res, sigma2=sigma2)

def fit_arx_ridge(u: np.ndarray, y: np.ndarray, na: int, nb: int, nk: int = 0, alpha: float = 1e-3) -> dict[str, Any]:
    Phi, Y, k0 = build_arx_phi(u, y, na, nb, nk)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(Phi, Y)
    theta = model.coef_.astype(float)
    a = theta[:na].copy()
    b = theta[na:].copy() if nb > 0 else np.array([])
    y_pred = Phi @ theta
    res = Y - y_pred
    sigma2 = float(np.var(res, ddof=max(1, Phi.shape[1])))
    return dict(a=a, b=b, theta=theta, k0=k0, res=res, sigma2=sigma2, alpha=alpha)

# -----------------------
# Companion / observer canonical conversion
# -----------------------
def arx_to_statespace(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ARX (y[k] = a1 y[k-1] + ... + ana y[k-na] + b0 u[k] + b1 u[k-1] + ...)
    into an observer-canonical state-space:
      x = [y[k-1], y[k-2], ..., y[k-na], u[k-1], u[k-2], ..., u[k-nb+1]]^T
    This is compatible with the build in previous script but clearer and numerically stable.
    """
    na = int(len(a))
    nb = int(len(b))
    nu_states = max(nb - 1, 0)
    nx = na + nu_states
    A = np.zeros((nx, nx), dtype=float)
    # top row: a and b_tail
    A[0, 0:na] = a
    if nu_states > 0:
        b_tail = b[1:]
        if len(b_tail) < nu_states:
            b_tail = np.pad(b_tail, (0, nu_states - len(b_tail)), 'constant')
        A[0, na:na + nu_states] = b_tail[:nu_states]
    # shift for y-history
    if na > 1:
        A[1:na, 0:na - 1] = np.eye(na - 1)
    # shift for u-history states
    if nu_states > 0:
        if nu_states > 1:
            A[na + 1:na + nu_states, na:na + nu_states - 1] = np.eye(nu_states - 1)
    B = np.zeros((nx, 1), dtype=float)
    B[0, 0] = b[0] if nb >= 1 else 0.0
    if nu_states > 0:
        B[na, 0] = 1.0
    C = np.zeros((1, nx), dtype=float)
    C[0, 0:na] = a
    if nu_states > 0:
        b_tail = b[1:]
        if len(b_tail) < nu_states:
            b_tail = np.pad(b_tail, (0, nu_states - len(b_tail)), 'constant')
        C[0, na:na + nu_states] = b_tail[:nu_states]
    D = np.array([[b[0] if nb >= 1 else 0.0]], dtype=float)
    return A, B, C, D

# -----------------------
# Simulation
# -----------------------
def simulate_ss(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, u: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    N = len(u)
    nx = A.shape[0]
    x = np.zeros((nx, 1), dtype=float) if x0 is None else x0.reshape((nx, 1)).astype(float)
    y_sim = np.zeros(N, dtype=float)
    for k in range(N):
        uk = np.array([[u[k]]], dtype=float)
        yk = (C @ x + D @ uk).item()
        y_sim[k] = yk
        x = A @ x + B @ uk
    return y_sim

# -----------------------
# Model selection: grid search (time-series CV) for na,nb and alpha
# -----------------------
def select_model_grid(u: np.ndarray, y: np.ndarray, method: str = 'ridge-arx', max_na: int = 6, max_nb: int = 4,
                      nk: int = 0, alphas: list[float] | None = None, cv_splits: int = 5):
    if alphas is None:
        alphas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best = None
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    N = len(y)
    for na in range(1, max_na + 1):
        for nb in range(1, max_nb + 1):
            for alpha in (alphas if method == 'ridge-arx' else [0.0]):
                rmses = []
                for train_idx, test_idx in tscv.split(np.arange(N)):
                    # construction requires contiguous sequences; use masks but keep indices contiguous by min/max
                    # We'll create u_train,y_train using train_idx; for robust TS-CV we require enough points for building Phi
                    u_tr, y_tr = u[train_idx], y[train_idx]
                    if len(y_tr) < max(na, nk + nb) + 5:
                        rmses.append(np.inf)
                        continue
                    if method == 'ridge-arx':
                        fit = fit_arx_ridge(u_tr, y_tr, na, nb, nk, alpha=alpha)
                    else:
                        fit = fit_arx_ols(u_tr, y_tr, na, nb, nk)
                    # simulate on test indices: need to simulate from start of test segment with initial states set from last na samples of train
                    test_start = test_idx[0]
                    # build initial states from the available history in combined signals
                    # for simplicity, simulate over full u[test_start:test_end] with initial x built from measured y,u
                    # construct companion SS
                    A, B, C, D = arx_to_statespace(fit['a'], fit['b'])
                    # prepare initial state from available past samples: y[test_start-1]..y[test_start-na], u similarly
                    def get_initial_x(k_start):
                        na_local = len(fit['a'])
                        nb_local = len(fit['b'])
                        nu_states = max(nb_local - 1, 0)
                        x0 = np.zeros((na_local + nu_states,), dtype=float)
                        for i in range(1, na_local + 1):
                            idx = k_start - i
                            x0[i - 1] = y[idx] if idx >= 0 else 0.0
                        for j in range(1, nu_states + 1):
                            idx = k_start - j
                            x0[na_local + j - 1] = u[idx] if idx >= 0 else 0.0
                        return x0
                    x0 = get_initial_x(test_start)
                    u_test = u[test_idx]
                    y_test = y[test_idx]
                    y_sim = simulate_ss(A, B, C, D, u_test, x0=x0)
                    rmse = math.sqrt(mean_squared_error(y_test, y_sim))
                    rmses.append(rmse)
                avg_rmse = np.mean([r for r in rmses if np.isfinite(r)]) if len(rmses) > 0 else np.inf
                if best is None or float(avg_rmse) < float(best['score']):
                    best = dict(method=method, na=na, nb=nb, nk=nk, alpha=alpha, score=avg_rmse)
    return best

# -----------------------
# Bootstrap residual resampling for parameter CI
# -----------------------
def bootstrap_arx(u: np.ndarray, y: np.ndarray, na: int, nb: int, nk: int, fit_func, n_iter: int = 200, random_state: int | None = None):
    rng = np.random.default_rng(random_state)
    fit0 = fit_func(u, y, na, nb, nk)
    theta0 = fit0['theta']
    Phi, Y, k0 = build_arx_phi(u, y, na, nb, nk)
    y_pred = Phi @ theta0
    resid = Y - y_pred
    thetas = np.zeros((n_iter, theta0.size), dtype=float)
    N = len(Y)
    for it in range(n_iter):
        resample = rng.choice(resid, size=N, replace=True)
        Y_boot = y_pred + resample
        # Refit by LS (unregularized) to the bootstrap target; for ridge we could keep same alpha but simpler to LS
        theta_b, *_ = np.linalg.lstsq(Phi, Y_boot, rcond=None)
        thetas[it, :] = theta_b
    mean_theta = thetas.mean(axis=0)
    ci_low = np.percentile(thetas, 2.5, axis=0)
    ci_high = np.percentile(thetas, 97.5, axis=0)
    return dict(mean=mean_theta, ci_low=ci_low, ci_high=ci_high, samples=thetas, fit0=fit0)

# -----------------------
# Train/Test Split for Time Series
# -----------------------
def split_train_test(u: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Split time series data into train and test sets.
    Returns: u_train, y_train, u_test, y_test, split_index
    """
    N = len(u)
    split_idx = int(N * (1.0 - test_size))
    split_idx = max(split_idx, 10)  # Ensure at least 10 samples in train
    split_idx = min(split_idx, N - 5)  # Ensure at least 5 samples in test

    u_train = u[:split_idx]
    y_train = y[:split_idx]
    u_test = u[split_idx:]
    y_test = y[split_idx:]

    return u_train, y_train, u_test, y_test, split_idx

# -----------------------
# Main CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced ARX -> State-space estimator with model selection & bootstrap")
    parser.add_argument('--file', '-f', default='experiment_data.csv')
    parser.add_argument('--method', choices=['arx', 'ridge-arx'], default='ridge-arx')
    parser.add_argument('--max-na', type=int, default=6)
    parser.add_argument('--max-nb', type=int, default=4)
    parser.add_argument('--order', type=int, default=None, help='Manually specify model order na (if provided, skips grid search).')
    parser.add_argument('--nk', type=int, default=0, help='Input delay (samples). Default 0 (u[k]).')
    parser.add_argument('--alphas', nargs='*', type=float, default=None, help='List of ridge alphas to search.')
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=0, help='Number of bootstrap iterations to estimate parameter CIs (0=off)')
    parser.add_argument('--detrend', action='store_true', help='Remove linear trend and mean before identification.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to use for testing (default 0.2)')
    parser.add_argument('--plot', action='store_true', help='Plot diagnostics (requires matplotlib).')
    args = parser.parse_args()

    u_raw, y_raw = read_data(args.file)
    u, y, info = detrend_remove_mean(u_raw.copy(), y_raw.copy(), detrend=args.detrend)

    # Train/Test split
    u_train, y_train, u_test, y_test, split_idx = split_train_test(u, y, test_size=args.test_size)
    print(f"Data split: {len(u_train)} training samples, {len(u_test)} test samples (split at index {split_idx})")

    # Model selection (using training data only)
    if args.order is None:
        best = select_model_grid(u_train, y_train, method=args.method, max_na=args.max_na, max_nb=args.max_nb,
                                 nk=args.nk, alphas=args.alphas, cv_splits=args.cv_splits)
        na = int(best['na'] if best is not None else 2)
        nb = int(best['nb'] if best is not None else 1)
        alpha = float(best.get('alpha', 0.0) if best is not None else 0.0)
    else:
        na = int(args.order)
        nb = max(1, min(args.max_nb, 1))  # default to 1 if not tuned
        alpha = (args.alphas[0] if args.alphas else 0.0)

    # Fit final model on training data only
    fit_func = (lambda uu, yy, na_, nb_, nk_: fit_arx_ridge(uu, yy, na_, nb_, nk_, alpha)) if args.method == 'ridge-arx' else fit_arx_ols
    fit = fit_func(u_train, y_train, na, nb, args.nk)

    # Build state-space realization
    A, B, C, D = arx_to_statespace(fit['a'], fit['b'])

    # Simulate on training data
    def build_x0_from_history(k_start, u_data, y_data):
        na_local = len(fit['a'])
        nb_local = len(fit['b'])
        nu_states_local = max(nb_local - 1, 0)
        x0 = np.zeros((na_local + nu_states_local,), dtype=float)
        for i in range(1, na_local + 1):
            idx = k_start - i
            x0[i - 1] = y_data[idx] if idx >= 0 else 0.0
        for j in range(1, nu_states_local + 1):
            idx = k_start - j
            x0[na_local + j - 1] = u_data[idx] if idx >= 0 else 0.0
        return x0

    x0_train = build_x0_from_history(fit['k0'], u_train, y_train)
    y_sim_train = simulate_ss(A, B, C, D, u_train, x0=x0_train)

    # Compute training residuals and RMSE
    valid_slice_train = slice(fit['k0'], len(y_train))
    y_train_valid = y_train[valid_slice_train]
    y_sim_train_valid = y_sim_train[valid_slice_train]
    rmse_train = math.sqrt(mean_squared_error(y_train_valid, y_sim_train_valid))

    # Simulate on test data (use last states from training or initial states from test boundary)
    x0_test = build_x0_from_history(0, u, y)  # Use combined data for initial state at split point
    # Adjust: use the actual split boundary states
    x0_test = np.zeros((A.shape[0],), dtype=float)
    na_local = len(fit['a'])
    nb_local = len(fit['b'])
    nu_states_local = max(nb_local - 1, 0)
    for i in range(1, na_local + 1):
        idx = split_idx - i
        x0_test[i - 1] = y[idx] if idx >= 0 else 0.0
    for j in range(1, nu_states_local + 1):
        idx = split_idx - j
        x0_test[na_local + j - 1] = u[idx] if idx >= 0 else 0.0

    y_sim_test = simulate_ss(A, B, C, D, u_test, x0=x0_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_sim_test))

    # Parameter bootstrap (on training data)
    boot_result = None
    if args.bootstrap and args.bootstrap > 1:
        boot_result = bootstrap_arx(u_train, y_train, na, nb, args.nk, fit_func, n_iter=args.bootstrap)
        # compute 95% CI for theta vector
        ci_low = boot_result['ci_low']
        ci_high = boot_result['ci_high']

    # Print concise report
    np.set_printoptions(precision=6, suppress=True)
    print("\nIdentification report")
    print("---------------------")
    print(f"File: {args.file}")
    print(f"Method: {args.method}")
    print(f"Selected na={na}, nb={nb}, nk={args.nk}, alpha={alpha}")
    print(f"Training data length: N={len(y_train)}; regression started at k={fit['k0']}")
    print(f"Test data length: N={len(y_test)}")
    print(f"RMSE (training): {rmse_train:.6g}")
    print(f"RMSE (test): {rmse_test:.6g}")
    print("")
    print("Estimated parameters (theta = [a1..ana, b0..b{nb-1}]):")
    print(f" theta: {fit['theta']}")
    if boot_result is not None:
        print(" Parameter 95% bootstrap CI (low, high):")
        for i, (low, high) in enumerate(zip(ci_low, ci_high)):
            print(f"  th[{i}] : [{low:.6g}, {high:.6g}]")
    print("")
    print("State-space realization (discrete-time observer-canonical):")
    print(f" A (shape {A.shape}):\n{A}")
    print(f" B (shape {B.shape}):\n{B}")
    print(f" C (shape {C.shape}):\n{C}")
    print(f" D (shape {D.shape}):\n{D}")

    # Poles and stability
    try:
        eigs = np.linalg.eigvals(A)
        stable = np.all(np.abs(eigs) < 1.0 + 1e-12)
        print(f"\nEigenvalues (poles) of A:\n {eigs}")
        print(f"All poles inside unit circle (stable): {stable}")
    except Exception as e:
        print(f"Could not compute eigenvalues: {e}")

    # Create control StateSpace object
    ss_sys = control.ss(A, B, C, D, True)
    print("\n(control) StateSpace object created.")

    # Plot diagnostics
    if args.plot:
        t = np.arange(len(y))
        t_train = t[:split_idx]
        t_test = t[split_idx:]

        # Combine simulations for full plot
        y_sim_full = np.concatenate([y_sim_train, y_sim_test])

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Full comparison with train/test regions
        axes[0].plot(t, y_raw, label='Measured', alpha=0.7)
        axes[0].plot(t, y_sim_full + info.get('y_mean', 0.0), '--', label='Simulated', linewidth=2)
        axes[0].axvline(fit['k0'], color='gray', linestyle=':', alpha=0.5, label='regression start')
        axes[0].axvline(split_idx, color='red', linestyle='--', linewidth=2, label='train/test split')
        axes[0].set_ylabel('Angle')
        axes[0].legend()
        axes[0].set_title('Full Signal: Measured vs Simulated')

        # Plot 2: Input signal
        axes[1].plot(t, u_raw, label='Input', color='green')
        axes[1].axvline(split_idx, color='red', linestyle='--', linewidth=2, label='train/test split')
        axes[1].set_ylabel('Input')
        axes[1].legend()

        # Plot 3: Training residuals
        axes[2].plot(t_train[fit['k0']:], y_train_valid - y_sim_train_valid, label=f'Train Residual (RMSE={rmse_train:.4g})', color='blue')
        axes[2].axhline(0, color='k', linestyle=':')
        axes[2].axvline(split_idx, color='red', linestyle='--', linewidth=2)
        axes[2].set_ylabel('Train Residual')
        axes[2].legend()

        # Plot 4: Test residuals
        axes[3].plot(t_test, y_test - y_sim_test, label=f'Test Residual (RMSE={rmse_test:.4g})', color='orange')
        axes[3].axhline(0, color='k', linestyle=':')
        axes[3].set_xlabel('sample k')
        axes[3].set_ylabel('Test Residual')
        axes[3].legend()

        plt.tight_layout()
        plt.savefig('estimation_diagnostics.png')
        plt.show()

    # Save result summary to file
    out_summary = {
        'method': args.method,
        'na': int(na), 'nb': int(nb), 'nk': int(args.nk), 'alpha': float(alpha),
        'theta': fit['theta'].tolist(),
        'A': A.tolist(), 'B': B.tolist(), 'C': C.tolist(), 'D': D.tolist(),
        'rmse_train': float(rmse_train),
        'rmse_test': float(rmse_test),
        'k0': int(fit['k0']),
        'split_idx': int(split_idx),
        'n_train': int(len(u_train)),
        'n_test': int(len(u_test))
    }
    summary_fn = os.path.splitext(args.file)[0] + '_id_summary.json'
    try:
        import json
        with open(summary_fn, 'w') as fh:
            json.dump(out_summary, fh, indent=2)
        print(f"\nSaved identification summary to: {summary_fn}")
    except Exception:
        pass

if __name__ == '__main__':
    main()
