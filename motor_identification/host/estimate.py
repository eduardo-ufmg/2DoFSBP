import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats
import sys

# --- Configuration ---
INPUT_FILENAME = 'experiment_data.csv'

# --- Hardcoded Physics Constants ---
# Shape: Ring (Thick-walled cylinder)
MASS = 0.09          # kg
RADIUS_INNER = 0.05  # meters
RADIUS_OUTER = 0.06  # meters

def main():
    print("--- Motor Parameter Estimator ---")

    # 1. Read experiment data
    try:
        print(f"Reading data from {INPUT_FILENAME}...")
        df = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILENAME}. Run experiment.py first.")
        return

    # Check if required columns exist
    required_cols = ['Time(s)', 'Input', 'Angle']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing columns. Found: {df.columns}. Expected: {required_cols}")
        return

    # 2. Clean data & Compute Derivatives (Velocity, Acceleration)
    print("Processing data (computing velocity and acceleration)...")
    
    # Calculate time step (dt)
    dt = df['Time(s)'].diff().mean()

    # Parameters for Savitzky-Golay filter
    # window_length: 11 samples (~110ms)
    window_length = 11 
    poly_order = 3

    # Calculate Velocity (1st derivative)
    df['Velocity'] = savgol_filter(df['Angle'], window_length, poly_order, deriv=1, delta=dt)

    # Calculate Acceleration (2nd derivative)
    df['Acceleration'] = savgol_filter(df['Angle'], window_length, poly_order, deriv=2, delta=dt)

    # 3. Physics Calculation (Ring)
    print("\n--- System Identification Setup ---")
    print(f"Load Properties: Ring (Mass={MASS}kg, R_in={RADIUS_INNER}m, R_out={RADIUS_OUTER}m)")

    # Moment of Inertia for a thick-walled ring: I = 0.5 * M * (R_inner^2 + R_outer^2)
    inertia = 0.5 * MASS * (RADIUS_INNER**2 + RADIUS_OUTER**2)
    print(f"Calculated Moment of Inertia (I): {inertia:.6e} kg*m^2")

    # Calculate Estimated Torque: Tau = I * alpha
    df['Estimated_Torque'] = inertia * df['Acceleration']

    # 4. Estimate Transfer Function (Linear Regression)
    # Model: Torque = K * Input_Signal + Offset
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['Input'], df['Estimated_Torque'])
    
    transfer_function_str = f"Torque(N*m) = {slope:.4f} * Input_Signal + {intercept:.4f}"
    
    print("\n--- Results ---")
    print(f"Transfer Function Estimate: {transfer_function_str}")
    print(f"Correlation (R-squared): {r_value**2:.4f}") # type: ignore

    # 5. Plotting
    print("Plotting results...")
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Input Signal
    axs[0].plot(df['Time(s)'], df['Input'], color='tab:blue')
    axs[0].set_ylabel('Input Signal\n(-1.0 to 1.0)')
    axs[0].set_title(f'System Response Analysis\nEst. TF: {transfer_function_str}')
    axs[0].grid(True, alpha=0.5)

    # Plot 2: Position (Angle)
    axs[1].plot(df['Time(s)'], df['Angle'], color='tab:orange')
    axs[1].set_ylabel('Position\n(Angle)')
    axs[1].grid(True, alpha=0.5)

    # Plot 3: Velocity
    axs[2].plot(df['Time(s)'], df['Velocity'], color='tab:green')
    axs[2].set_ylabel('Velocity\n(rad/s)')
    axs[2].grid(True, alpha=0.5)

    # Plot 4: Acceleration (which maps directly to Torque)
    axs[3].plot(df['Time(s)'], df['Acceleration'], color='tab:red')
    axs[3].set_ylabel('Acceleration\n(rad/s^2)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('estimation_results.png')
    print("Plot saved to 'estimation_results.png'. Displaying now...")
    plt.show()

if __name__ == "__main__":
    main()