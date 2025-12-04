import argparse
import pandas as pd
import numpy as np

# Physical constants
KB = 1.380649e-23       # Boltzmann constant (J/K)
H = 6.62607015e-34      # Planck constant (J·s)
R_KCAL = 1.987204259e-3 # Gas constant (kcal/(mol·K))
KB_H = KB / H           # Pre-calculated ratio (s⁻¹·K⁻¹)

def calculate_eyring_k(barrier, T):
    """
    Calculate rate constant k using the Eyring equation.
    k = (kB * T / h) * exp(-barrier / (R * T))
    
    Args:
        barrier (float or np.array): Activation energy in kcal/mol.
        T (float): Temperature in Kelvin.
        
    Returns:
        float or np.array: Rate constant k in s⁻¹.
    """
    prefactor = KB_H * T
    exponent = -barrier / (R_KCAL * T)
    return prefactor * np.exp(exponent)

def main():
    parser = argparse.ArgumentParser(
        description="Compute rate constants (k) from forward and reverse energy barriers."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input CSV file path. Must contain 3 columns: Step ID, Forward Barrier (kcal/mol), Reverse Barrier (kcal/mol)."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output CSV file path."
    )
    parser.add_argument(
        "-t", "--temp", type=float, default=298.15,
        help="Temperature in Kelvin (default: 298.15)."
    )

    args = parser.parse_args()

    try:
        # Read the CSV file
        # We don't rely on header names, just positions (0, 1, 2)
        df = pd.read_csv(args.input)
        
        if df.shape[1] < 3:
            raise ValueError("Input CSV must have at least 3 columns.")

        # Extract barriers (assuming column 1 is forward, column 2 is reverse)
        # Column 0 is the step name/ID.
        fwd_barriers = df.iloc[:, 1].astype(float)
        rev_barriers = df.iloc[:, 2].astype(float)

        # Calculate rate constants
        k_fwd = calculate_eyring_k(fwd_barriers, args.temp)
        k_rev = calculate_eyring_k(rev_barriers, args.temp)

        # Prepare output data (Wide format: single row)
        num_steps = len(df)
        
        # Generate column names: k1, k2, ... and k-1, k-2, ...
        # Note: Using 1-based indexing to match standard chemical notation conventions implied by "k1, k2"
        cols_fwd = [f"k{i+1}" for i in range(num_steps)]
        cols_rev = [f"k-{i+1}" for i in range(num_steps)]
        
        all_cols = cols_fwd + cols_rev
        
        # Concatenate values into a single array
        all_values = np.concatenate([k_fwd.values, k_rev.values])
        
        # Create output DataFrame (1 row)
        df_out = pd.DataFrame([all_values], columns=all_cols)
        
        # Save to CSV
        df_out.to_csv(args.output, index=False)
        
        print(f"Successfully processed {num_steps} steps.")
        print(f"Temperature: {args.temp} K")
        print(f"Output saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
