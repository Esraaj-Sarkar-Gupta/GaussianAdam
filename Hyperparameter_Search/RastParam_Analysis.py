import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def analyze_hyperparams(csv_path="params.csv"):
    # ---- Load the Data ---- #
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: params.csv not found. Please ensure the file exists.")
        return

    # ---- Find the Absolute Best Combination --- #
    min_idx = df['final_f'].idxmin()
    best_row = df.iloc[min_idx]

    print("\n" + "="*40)
    print("BEST HYPERPARAMETER COMBINATION")
    print("="*40)
    print(f"Alpha (Learning Rate): {best_row['alpha']}")
    print(f"Gamma (Variance Decay):{best_row['gamma']}")
    print(f"Std Dev (Initial Noise):{best_row['std']}")
    print(f"Lowest Final Loss:     {best_row['final_f']:.6f}")
    print("="*40 + "\n")

    # 3. Setup the Plots
    # We have 10 std values, so a 2x5 grid is perfect
    unique_stds = np.sort(df['std'].unique())
    
    if len(unique_stds) == 0:
        print("No data found in CSV.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10), constrained_layout=True)
    axes = axes.flatten() # Flatten 2D array to 1D loop

    vmin = df['final_f'].min()
    vmax = df['final_f'].quantile(0.90) # Cap color at 90th percentile to ignore massive outliers

    print("Generating contour plots...")

    for i, std_val in enumerate(unique_stds):
        if i >= len(axes): break # Safety check
        
        ax = axes[i]
        
        # Filter data for this specific STD
        subset = df[df['std'] == std_val]
        
        cntr = ax.tricontourf(
            subset['gamma'], 
            subset['alpha'], 
            subset['final_f'], 
            levels=20, 
            cmap='viridis_r', # Reversed: Purple/Dark = Low Loss (Good), Yellow = High Loss (Bad)
            vmin=vmin,
            vmax=vmax
        )
        
        # Plot styling
        ax.set_title(f"Initial Noise (std) = {std_val}")
        ax.set_xlabel("Gamma (Decay)")
        ax.set_ylabel("Alpha (LR)")
        
        # Add a star marker if this plot contains the Global Best
        if std_val == best_row['std']:
            ax.plot(best_row['gamma'], best_row['alpha'], 'r*', markersize=20, markeredgecolor='white', label='Global Best')
            ax.legend(loc='upper right')

        # Add colorbar for this specific plot
        fig.colorbar(cntr, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'GaussianAdam Hyperparameter Search (Loss Landscape)\nGlobal Best Loss: {best_row["final_f"]:.4f}', fontsize=16)
    
    # Save and Show
    #plt.savefig("hyperparam_search_results.png", dpi=150)
    print("Plot saved to 'hyperparam_search_results.png'")
    plt.show()

if __name__ == "__main__":
    analyze_hyperparams()