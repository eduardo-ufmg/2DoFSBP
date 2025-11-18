import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import linalg

import matplotlib.pyplot as plt


def read_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Read input and output data from CSV file."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    df = pd.read_csv(filename)
    if 'Input' not in df.columns or 'Angle' not in df.columns:
        raise ValueError("CSV must contain columns named 'Input' and 'Angle'.")
    u = df['Input'].astype(float).to_numpy()
    y = df['Angle'].astype(float).to_numpy()
    # Remove NaNs
    mask = ~(np.isnan(u) | np.isnan(y))
    if not mask.all():
        u = u[mask]
        y = y[mask]
    return u, y


def load_identified_model(summary_file: str) -> dict:
    """Load identified model matrices from JSON summary."""
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Model summary file not found: {summary_file}")
    with open(summary_file, 'r') as fh:
        summary = json.load(fh)
    A = np.array(summary['A'], dtype=float)
    B = np.array(summary['B'], dtype=float)
    C = np.array(summary['C'], dtype=float)
    D = np.array(summary['D'], dtype=float)
    return {'A': A, 'B': B, 'C': C, 'D': D, 'summary': summary}


def kalman_filter(A, B, C, D, u, y, Q, R, x0=None, P0=None):
    """
    Discrete-time Kalman filter for state estimation.

    System model:
        x[k+1] = A*x[k] + B*u[k] + w[k]    (process noise w ~ N(0, Q))
        y[k]   = C*x[k] + D*u[k] + v[k]    (measurement noise v ~ N(0, R))

    Args:
        A, B, C, D: State-space matrices
        u: Input sequence (N,)
        y: Output measurement sequence (N,)
        Q: Process noise covariance (n_states x n_states)
        R: Measurement noise covariance (scalar or 1x1)
        x0: Initial state estimate (n_states,)
        P0: Initial error covariance (n_states x n_states)

    Returns:
        x_est: Estimated states (N x n_states)
        P_hist: Error covariance history (N x n_states x n_states)
        innovations: Innovation sequence (N,)
    """
    N = len(u)
    nx = A.shape[0]

    # Initialize
    if x0 is None:
        x0 = np.zeros((nx, 1))
    else:
        x0 = x0.reshape((nx, 1))

    if P0 is None:
        P0 = np.eye(nx) * 1.0

    x_est = np.zeros((N, nx))
    P_hist = np.zeros((N, nx, nx))
    innovations = np.zeros(N)

    x_pred = x0.copy()
    P_pred = P0.copy()

    R_scalar = float(R[0, 0])

    for k in range(N):
        uk = np.array([[u[k]]])
        yk_meas = y[k]

        # Prediction
        yk_pred = (C @ x_pred + D @ uk).item()
        innovation = yk_meas - yk_pred
        innovations[k] = innovation

        # Innovation covariance
        S = (C @ P_pred @ C.T).item() + R_scalar

        # Kalman gain
        K = (P_pred @ C.T) / S

        # Update
        x_est_k = x_pred + K * innovation
        P_est_k = (np.eye(nx) - K @ C) @ P_pred

        # Store
        x_est[k, :] = x_est_k.flatten()
        P_hist[k, :, :] = P_est_k

        # Predict next
        x_pred = A @ x_est_k + B @ uk
        P_pred = A @ P_est_k @ A.T + Q

    return x_est, P_hist, innovations


def steady_state_kalman_filter(A, B, C, D, u, y, Q, R, x0=None):
    """
    Kalman filter using steady-state Kalman gain (faster).
    Solves the discrete-time algebraic Riccati equation (DARE) for steady-state P.
    """
    N = len(u)
    nx = A.shape[0]

    # Solve DARE for steady-state P
    try:
        P_ss = linalg.solve_discrete_are(A.T, C.T, Q, R)
    except Exception as e:
        print(f"Warning: Could not solve DARE, using time-varying filter. Error: {e}")
        x_est, P_hist, innovations = kalman_filter(A, B, C, D, u, y, Q, R, x0)
        # Return with dummy values for P_ss and K_ss to match expected 4-tuple
        return x_est, P_hist[-1], innovations, None

    # Steady-state Kalman gain
    R_scalar = float(R[0, 0])
    S_ss = (C @ P_ss @ C.T).item() + R_scalar
    K_ss = (P_ss @ C.T) / S_ss

    # Initialize
    if x0 is None:
        x0 = np.zeros((nx, 1))
    else:
        x0 = x0.reshape((nx, 1))

    x_est = np.zeros((N, nx))
    innovations = np.zeros(N)

    x_pred = x0.copy()

    for k in range(N):
        uk = np.array([[u[k]]])
        yk_meas = y[k]

        # Prediction
        yk_pred = (C @ x_pred + D @ uk).item()
        innovation = yk_meas - yk_pred
        innovations[k] = innovation

        # Update
        x_est_k = x_pred + K_ss * innovation

        # Store
        x_est[k, :] = x_est_k.flatten()

        # Predict next
        x_pred = A @ x_est_k + B @ uk

    return x_est, P_ss, innovations, K_ss


def main():
    parser = argparse.ArgumentParser(description="State estimation using identified ARX model")
    parser.add_argument('--data', '-d', default='experiment_data.csv', help='Data file with Input and Angle columns')
    parser.add_argument('--model', '-m', default='experiment_data_id_summary.json', help='Identified model JSON file')
    parser.add_argument('--Q', type=float, default=1e-6, help='Process noise covariance (scalar, will create Q=q*I)')
    parser.add_argument('--R', type=float, default=1e-4, help='Measurement noise covariance (scalar)')
    parser.add_argument('--steady-state', action='store_true', help='Use steady-state Kalman gain (faster)')
    parser.add_argument('--plot', action='store_true', help='Plot state estimates and innovations')
    args = parser.parse_args()

    # Load data
    u, y = read_data(args.data)
    print(f"Loaded {len(u)} samples from {args.data}")

    # Load identified model
    model = load_identified_model(args.model)
    A, B, C, D = model['A'], model['B'], model['C'], model['D']
    nx = A.shape[0]
    print(f"Loaded model from {args.model}")
    print(f"State dimension: {nx}")

    # Construct noise covariance matrices
    Q = np.eye(nx) * args.Q
    R = np.array([[args.R]])

    # Run Kalman filter
    print(f"\nRunning Kalman filter (Q={args.Q}, R={args.R})...")
    if args.steady_state:
        x_est, P_ss, innovations, K_ss = steady_state_kalman_filter(A, B, C, D, u, y, Q, R)
        print(f"Steady-state error covariance P:\n{P_ss}")
        if K_ss is not None:
            print(f"Steady-state Kalman gain K:\n{K_ss.flatten()}")
    else:
        x_est, P_hist, innovations = kalman_filter(A, B, C, D, u, y, Q, R)
        print(f"Time-varying Kalman filter complete")

    # Compute reconstruction
    y_recon = np.zeros(len(u))
    for k in range(len(u)):
        y_recon[k] = (C @ x_est[k:k+1].T + D * u[k]).item()

    rmse_recon = np.sqrt(np.mean((y - y_recon)**2))
    print(f"\nReconstruction RMSE: {rmse_recon:.6g}")
    print(f"Innovation std: {np.std(innovations):.6g}")

    # Save state estimates
    state_df = pd.DataFrame(x_est, columns=[f'x{i+1}' for i in range(nx)])
    state_df['u'] = u
    state_df['y_meas'] = y
    state_df['y_recon'] = y_recon
    state_df['innovation'] = innovations

    out_file = os.path.splitext(args.data)[0] + '_states.csv'
    state_df.to_csv(out_file, index=False)
    print(f"Saved state estimates to {out_file}")

    # Plot
    if args.plot:
        t = np.arange(len(u))

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Output reconstruction
        axes[0].plot(t, y, label='Measured y', alpha=0.7)
        axes[0].plot(t, y_recon, '--', label='Reconstructed y', linewidth=2)
        axes[0].set_ylabel('Angle')
        axes[0].legend()
        axes[0].set_title(f'Kalman Filter State Estimation (RMSE={rmse_recon:.4g})')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: States
        for i in range(nx):
            axes[1].plot(t, x_est[:, i], label=f'x{i+1}', alpha=0.8)
        axes[1].set_ylabel('States')
        axes[1].legend(ncol=min(nx, 4))
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Input
        axes[2].plot(t, u, color='green', label='Input u')
        axes[2].set_ylabel('Input')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Innovations
        axes[3].plot(t, innovations, color='red', alpha=0.6, label=f'Innovation (std={np.std(innovations):.4g})')
        axes[3].axhline(0, color='k', linestyle=':', alpha=0.5)
        axes[3].set_xlabel('Sample k')
        axes[3].set_ylabel('Innovation')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.splitext(args.data)[0] + '_states.png'
        plt.savefig(plot_file, dpi=150)
        print(f"Saved plot to {plot_file}")
        plt.show()


if __name__ == '__main__':
    main()