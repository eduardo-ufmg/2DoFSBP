import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.linalg import lstsq  # Using least squares for multiple regression
import json

# --- Configuration ---
INPUT_FILENAME = 'experiment_data.csv'
OUTPUT_MODEL_FILE = 'model_parameters.json'

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

    # Check columns
    required_cols = ['Time(s)', 'Input', 'Angle']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing columns. Found: {df.columns}. Expected: {required_cols}")
        return

    # 2. Clean data & Compute Derivatives
    print("Processing data...")
    dt = df['Time(s)'].diff().mean()
    
    # Savitzky-Golay filter settings
    window_length = 11 
    poly_order = 3

    # We need a longer window for stable derivatives, especially acceleration
    # Let's increase it.
    window_length_deriv = 51 # Must be odd
    if window_length_deriv <= poly_order:
        print(f"Warning: window_length ({window_length_deriv}) <= poly_order ({poly_order}). Adjusting.")
        window_length_deriv = poly_order + 2
        if window_length_deriv % 2 == 0: # ensure odd
             window_length_deriv += 1

    print(f"Using Sav-Gol filter: window={window_length_deriv}, order={poly_order}")
    df['Velocity'] = savgol_filter(df['Angle'], window_length_deriv, poly_order, deriv=1, delta=dt)
    df['Acceleration'] = savgol_filter(df['Angle'], window_length_deriv, poly_order, deriv=2, delta=dt)

    # 3. Physics Calculation (for Torque TF)
    print(f"Load Properties: Ring (Mass={MASS}kg, R_in={RADIUS_INNER}m, R_out={RADIUS_OUTER}m)")
    
    # Inertia for thick-walled ring
    inertia = 0.5 * MASS * (RADIUS_INNER**2 + RADIUS_OUTER**2)
    print(f"Calculated Moment of Inertia (I): {inertia:.6e} kg*m^2")

    # 4. Estimate Dynamic Transfer Function
    # We are fitting the model:
    # Velocity = K_m * Input - T_m * Acceleration + Intercept
    
    # We use numpy.linalg.lstsq to solve A*x = b
    # b is the dependent variable (Velocity)
    b = df['Velocity'].to_numpy(dtype=float)
    
    # A contains the independent variables (Input, Acceleration, and a constant)
    A = np.vstack([
        df['Input'].to_numpy(dtype=float), 
        df['Acceleration'].to_numpy(dtype=float), 
        np.ones(len(df))
    ]).T
    
    # Solve for x = [coeff_Input, coeff_Accel, coeff_Intercept]
    try:
        lstsq_res = lstsq(A, b)
        assert lstsq_res is not None, "Least squares solution failed."
        x, residuals, rank, s = lstsq_res 

        # Extract parameters
        K_m = x[0]
        T_m = -x[1]  # Note the negative sign
        Intercept = x[2]

        # Calculate R^2 for fit
        ss_total = np.sum((b - np.mean(b))**2)

        # `residuals` returned by scipy.linalg.lstsq can be an empty array
        # (for underdetermined systems) or a scalar; handle both cases.
        try:
            res_arr = np.asarray(residuals)
            if res_arr.size == 0:
                # Compute residuals manually if not provided
                ss_residual = np.sum((b - A.dot(x))**2)
            else:
                ss_residual = float(res_arr.ravel()[0])
        except Exception:
            # Fallback: compute residuals directly
            ss_residual = np.sum((b - A.dot(x))**2)

        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    except Exception as e:
        print(f"Error during Linear Regression: {e}")
        return

    print("\n--- Model Estimation Results ---")
    print(f"Model: Velocity = K_m * Input - T_m * Acceleration + Intercept")
    print(f"  K_m (Gain):         {K_m:.6f}")
    print(f"  T_m (Time Const):   {T_m:.6f}")
    print(f"  Intercept:        {Intercept:.6f}")
    print(f"  Fit (R^2):        {r_squared:.6f}")

    # --- Derived Transfer Functions ---
    # G_p(s) = Angle(s) / Input(s) = K_m / (T_m*s^2 + s)
    tf_angle = f"G_p(s) = {K_m:.4f} / ({T_m:.4f}*s^2 + s)"
    
    # G_v(s) = Velocity(s) / Input(s) = s * G_p(s) = K_m / (T_m*s + 1)
    tf_velocity = f"G_v(s) = {K_m:.4f} / ({T_m:.4f}*s + 1)"

    # G_a(s) = Accel(s) / Input(s) = s * G_v(s) = {K_m*s} / (T_m*s + 1)
    tf_accel = f"G_a(s) = ({K_m:.4f}*s) / ({T_m:.4f}*s + 1)"

    # G_t(s) = Torque(s) / Input(s) = I * G_a(s) = (I*K_m*s) / (T_m*s + 1)
    tf_torque_gain = inertia * K_m
    tf_torque = f"G_t(s) = ({tf_torque_gain:.4e}*s) / ({T_m:.4f}*s + 1)"

    print("\n--- Derived Transfer Functions ---")
    print(f"Input -> Angle:       {tf_angle}")
    print(f"Input -> Velocity:    {tf_velocity}")
    print(f"Input -> Acceleration: {tf_accel}")
    print(f"Input -> Torque (Est): {tf_torque}")


    # 5. Save Model to JSON
    model_data = {
        "model_type": "Dynamic_2ndOrder",
        "model_equation": "T_m * Accel + Velocity = K_m * Input + Intercept",
        "parameters": {
            "K_m_gain": K_m,
            "T_m_time_constant": T_m,
            "intercept": Intercept,
            "r_squared": r_squared
        },
        "physics": {
            "inertia_kg_m2": inertia,
            "mass_kg": MASS,
            "radius_inner_m": RADIUS_INNER,
            "radius_outer_m": RADIUS_OUTER
        },
        "transfer_functions": {
            "input_to_angle": tf_angle,
            "input_to_velocity": tf_velocity,
            "input_to_acceleration": tf_accel,
            "input_to_torque": tf_torque
        }
    }
    
    try:
        with open(OUTPUT_MODEL_FILE, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"\n[SUCCESS] Model parameters saved to '{OUTPUT_MODEL_FILE}'.")
    except Exception as e:
        print(f"Error saving model file: {e}")

    # 6. Plotting
    print("Plotting results...")
    
    # Calculate predicted velocity for validation plot
    df['Predicted_Velocity'] = K_m * df['Input'] - T_m * df['Acceleration'] + Intercept
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    fig.suptitle(f'Dynamic Model Analysis (R^2 = {r_squared:.4f})', fontsize=16)

    # --- Plot 1: Input ---
    axs[0].plot(df['Time(s)'], df['Input'], color='tab:blue')
    axs[0].set_ylabel('Input Signal')
    axs[0].set_title(f'Input -> Angle: {tf_angle}')
    axs[0].grid(True, alpha=0.5)

    # --- Plot 2: Angle ---
    axs[1].plot(df['Time(s)'], df['Angle'], color='tab:orange')
    axs[1].set_ylabel('Position (Angle)')
    axs[1].grid(True, alpha=0.5)

    # --- Plot 3: Velocity (Measured vs. Predicted) ---
    axs[2].plot(df['Time(s)'], df['Velocity'], color='tab:green', label='Measured Velocity')
    axs[2].plot(df['Time(s)'], df['Predicted_Velocity'], color='tab:red', 
                linestyle='--', label='Predicted Velocity (from model)')
    axs[2].set_ylabel('Velocity (rad/s)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.5)
    axs[2].set_title(f'Input -> Velocity: {tf_velocity}')

    # --- Plot 4: Acceleration ---
    axs[3].plot(df['Time(s)'], df['Acceleration'], color='tab:purple')
    axs[3].set_ylabel('Acceleration (rad/s^2)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True, alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
    plt.savefig('estimation_results.png')
    print("Plot saved to 'estimation_results.png'. Displaying...")
    plt.show()

if __name__ == "__main__":
    main()