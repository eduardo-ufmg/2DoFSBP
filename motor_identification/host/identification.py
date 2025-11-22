import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILENAME = "experiment_data.csv"
J = 0.000153  # Moment of Inertia [kg.m^2]

# Filter settings (Adjust these if your data is very noisy)
WINDOW_LENGTH = 11  # Must be odd. Higher = smoother but creates lag
POLY_ORDER = 3  # Polynomial order for the filter


def identify_motor_dynamics():
    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    try:
        df = pd.read_csv(FILENAME)
        # Ensure headers define Time, Input, Angle (strip spaces just in case)
        df.columns = [c.strip() for c in df.columns]
    except FileNotFoundError:
        print(f"Error: Could not find file '{FILENAME}'.")
        return

    # Extract numpy arrays for faster processing
    t = df["Time"].to_numpy()
    u = df["Input"].to_numpy()
    theta = df["Angle"].to_numpy()  # radians

    # Calculate sampling time (dt) assuming constant frequency
    dt = float(np.mean(np.diff(t)))
    print(f"Data loaded. {len(df)} samples. Sampling time approx: {dt:.4f} s")

    # ==========================================
    # 3. PREPROCESSING (Differentiation)
    # ==========================================
    # We use Savitzky-Golay to smooth and differentiate simultaneously.
    # deriv=1 calculates velocity, deriv=2 calculates acceleration.

    # 1st Derivative: Angular Velocity (omega)
    omega = savgol_filter(
        theta, window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, deriv=1, delta=dt
    )

    # 2nd Derivative: Angular Acceleration (alpha)
    alpha = savgol_filter(
        theta, window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, deriv=2, delta=dt
    )

    # ==========================================
    # 4. PHYSICS CALCULATION (Inverse Dynamics)
    # ==========================================
    # Calculate the Net Torque required to produce the observed acceleration
    tau_net = J * alpha

    # ==========================================
    # 5. IDENTIFICATION (Regression)
    # ==========================================
    # Model Structure: J*alpha = (K * u) - (b * omega) + offset
    # Rearranged for regression: J*alpha = [u, omega, 1] * [K, -b, c]^T

    # Prepare Feature Matrix X: [Input, Velocity]
    # We stack 'u' and 'omega' as columns
    X = np.column_stack((u, omega))
    y = tau_net

    # Fit the linear model
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    K_est = model.coef_[0]  # Coefficient for u (Input Gain)
    b_est = -model.coef_[
        1
    ]  # Coefficient for omega (Damping/Friction), note the negative sign
    c_est = model.intercept_  # Bias/Offset

    # Calculate R-squared to check goodness of fit
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # ==========================================
    # 6. RESULTS & VISUALIZATION
    # ==========================================
    print("-" * 30)
    print("IDENTIFICATION RESULTS")
    print("-" * 30)
    print(f"Model Quality (R^2): {r2:.4f} (1.0 is perfect)")
    print(f"Input Gain (K):      {K_est:.6f} [Nm / unit_input]")
    print(f"Damping Coeff (b):   {b_est:.6f} [Nm / (rad/s)]")
    print(f"Constant Bias:       {c_est:.6f} [Nm]")
    print("-" * 30)
    print("FINAL DYNAMIC MODEL (u -> tau_m):")
    print(f"tau_m = {K_est:.4f} * u")
    print("(Note: The NET torque acting on the load is: tau_m - friction)")
    print("-" * 30)

    # Save npz and json
    np.savez("identified_motor_model.npz", K=K_est, b=b_est, c=c_est, J=J, r2=r2)
    with open("identified_motor_model.json", "w") as f:
        json.dump({"K": K_est, "b": b_est, "c": c_est, "J": J, "r2": r2}, f, indent=4)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot 1: Inputs
    plt.subplot(3, 1, 1)
    plt.plot(t, u, label="Input (u)", color="blue", alpha=0.7)
    plt.ylabel("Input [-1, 1]")
    plt.title("System Identification Data")
    plt.legend()
    plt.grid(True)

    # Plot 2: Kinematics (Computed Speed)
    plt.subplot(3, 1, 2)
    plt.plot(t, omega, label="Speed (rad/s)", color="orange")
    plt.ylabel("Speed [rad/s]")
    plt.legend()
    plt.grid(True)

    # Plot 3: Dynamics (Torque Match)
    plt.subplot(3, 1, 3)
    plt.plot(
        t, tau_net, label="Measured Net Torque (J*alpha)", color="black", alpha=0.5
    )
    plt.plot(
        t, y_pred, label="Model Prediction (K*u - b*w)", color="red", linestyle="--"
    )
    plt.ylabel("Torque [Nm]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    identify_motor_dynamics()
