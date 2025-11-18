import argparse
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def read_data(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    df = pd.read_csv(filename)
    if 'Input' not in df.columns or 'Angle' not in df.columns:
        raise ValueError("CSV must contain 'Input' and 'Angle' columns.")
    u = df['Input'].to_numpy(dtype=float)
    y = df['Angle'].to_numpy(dtype=float)
    if u.shape[0] != y.shape[0]:
        raise ValueError("Input and Angle must have same length.")
    return u, y

def fit_arx(u, y, na=2, nb=1):
    """
    Fit ARX model coefficients using least squares.

    Model used:
      y[k] = sum_{i=1..na} a_i * y[k-i] + sum_{j=0..nb-1} b_j * u[k-j] + e[k]

    Returns:
      a: array shape (na,) -> [a1, a2, ...]
      b: array shape (nb,) -> [b0, b1, ...]
      k_start: first k used in regression
    """
    N = len(y)
    if N < max(na, nb) + 5:
        # require some minimal data
        raise ValueError(f"Not enough data (N={N}) for na={na}, nb={nb}.")
    k_start = max(na, nb - 1)
    rows = []
    targets = []
    for k in range(k_start, N):
        phi = []
        # past outputs y[k-1] .. y[k-na]
        for i in range(1, na + 1):
            phi.append(y[k - i])
        # current and past inputs u[k], u[k-1], ...
        for j in range(0, nb):
            idx = k - j
            # if index out-of-range (shouldn't be due to k_start), use 0
            phi.append(u[idx] if idx >= 0 else 0.0)
        rows.append(phi)
        targets.append(y[k])
    Phi = np.vstack(rows)
    Y = np.array(targets)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    a = theta[0:na].astype(float)
    b = theta[na:na + nb].astype(float)
    return a, b, k_start

def build_state_space_from_arx(a, b):
    """
    Build a discrete-time state-space (A,B,C,D) from ARX coefficients with
    the ARX model convention used in fit_arx().

    State vector constructed as:
      x = [ y[k-1], y[k-2], ..., y[k-na],  u[k-1], u[k-2], ..., u[k-(nb-1)] ]^T

    Dimensions:
      na = len(a)
      nb = len(b)
      nx = na + max(nb-1, 0)

    Equations:
      y[k] = [a1 ... ana, b1 ... b_{nb-1}] * x + b0 * u[k]
      x_{k+1} = A x_k + B u[k]
    """
    na = len(a)
    nb = len(b)
    nu_states = max(nb - 1, 0)
    nx = na + nu_states

    A = np.zeros((nx, nx), dtype=float)
    B = np.zeros((nx, 1), dtype=float)
    # Top-left block (na x na)
    # First row: a1..ana
    A[0, 0:na] = a
    # Shift rows for output-history states
    if na > 1:
        A[1:na, 0:na - 1] = np.eye(na - 1)

    # Top-right block: coupling from past-input-states (b1..b_{nb-1})
    if nu_states > 0:
        # b[1:] corresponds to b1..b_{nb-1}
        b_tail = b[1:]
        # pad if lengths mismatch
        if b_tail.shape[0] < nu_states:
            b_tail = np.pad(b_tail, (0, nu_states - b_tail.shape[0]), 'constant')
        A[0, na:na + nu_states] = b_tail[:nu_states]

    # Bottom-right block: shift for past-input-states
    if nu_states > 0:
        # For u-state vector [u[k-1], u[k-2], ...], its update is:
        # new_u_states = [u[k], u[k-1], ...] -> shift with ones on subdiagonal
        for i in range(nu_states - 1):
            A[na + 1 + i, na + i] = 1.0

    # B: input influence
    # y equation gets b0 * u[k]
    B[0, 0] = b[0] if nb >= 1 else 0.0
    # the first u-state (if exists) becomes u[k] -> coefficient 1
    if nu_states > 0:
        B[na, 0] = 1.0

    # Output equation
    C = np.zeros((1, nx), dtype=float)
    # C multiplies x = [y[k-1]..y[k-na], u[k-1]..]
    C[0, 0:na] = a
    if nu_states > 0:
        # C includes b1..b_{nb-1}
        b_tail = b[1:]
        if b_tail.shape[0] < nu_states:
            b_tail = np.pad(b_tail, (0, nu_states - b_tail.shape[0]), 'constant')
        C[0, na:na + nu_states] = b_tail[:nu_states]

    D = np.array([[b[0] if nb >= 1 else 0.0]], dtype=float)

    return A, B, C, D

def simulate_ss(A, B, C, D, u, x0=None):
    """
    Simulate discrete-time state-space model:
      x_{k+1} = A x_k + B u_k
      y_k     = C x_k + D u_k

    Returns y_sim array same length as u.
    """
    N = len(u)
    nx = A.shape[0]
    x = np.zeros((nx, 1), dtype=float) if x0 is None else x0.reshape(nx, 1)
    y_sim = np.zeros(N, dtype=float)
    for k in range(N):
        uk = np.array([[u[k]]], dtype=float)
        yk = (C @ x + D @ uk).item()
        y_sim[k] = yk
        x = A @ x + B @ uk
    return y_sim

def main():
    parser = argparse.ArgumentParser(description="ARX -> State-space estimator for experiment_data.csv")
    parser.add_argument('--file', '-f', default='experiment_data.csv', help='CSV filename (default: experiment_data.csv)')
    parser.add_argument('--na', type=int, default=2, help='Number of past outputs (na), default 2')
    parser.add_argument('--nb', type=int, default=1, help='Number of input terms (nb), default 1 (includes u[k])')
    parser.add_argument('--plot', action='store_true', help='Plot measured vs simulated outputs (requires matplotlib)')
    args = parser.parse_args()

    u, y = read_data(args.file)
    na = max(1, int(args.na))
    nb = max(1, int(args.nb))

    # remove mean to avoid bias
    u = u - np.mean(u)
    y = y - np.mean(y)

    a, b, k_start = fit_arx(u, y, na=na, nb=nb)

    A, B, C, D = build_state_space_from_arx(a, b)

    # Print results
    np.set_printoptions(precision=6, suppress=True)
    print("Estimated ARX coefficients:")
    print(f"  a (na={na}): {a}")
    print(f"  b (nb={nb}): {b}")
    print("\nState-space realization (discrete-time):")
    print(f"  A (shape {A.shape}):\n{A}")
    print(f"  B (shape {B.shape}):\n{B}")
    print(f"  C (shape {C.shape}):\n{C}")
    print(f"  D (shape {D.shape}):\n{D}")

    # Simulate identified model on the input sequence
    y_sim = simulate_ss(A, B, C, D, u)

    # Compute simple goodness-of-fit on overlapping region used for training
    from math import sqrt
    valid_idx = slice(k_start, len(y))
    err = y[valid_idx] - y_sim[valid_idx]
    rmse = sqrt(np.mean(err**2)) if err.size > 0 else float('nan')
    print(f"\nData used for regression starts at sample k = {k_start}")
    print(f"RMSE between measured and simulated output (on regression range): {rmse:.6g}")

    if args.plot:
        t = np.arange(len(y))
        plt.figure(figsize=(9, 4))
        plt.plot(t, y, label='Measured (Angle)', linewidth=1.2)
        plt.plot(t, y_sim, label='Simulated (identified SS)', linestyle='--', linewidth=1.2)
        plt.axvline(k_start, color='gray', linestyle=':', label='regression start')
        plt.xlabel('sample k')
        plt.ylabel('Angle')
        plt.title('Measured vs Simulated Output')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
