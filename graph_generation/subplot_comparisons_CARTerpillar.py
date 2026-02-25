import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# --- Configuration ---
logs_folder = "logs"

environments = [
    "CARTerpillar9.81_6",
    "CARTerpillar9.81_7",
    "CARTerpillar9.81_8",
    "CARTerpillar9.81_9",
    "CARTerpillar9.81_10",
    "CARTerpillar9.81_11",
]

# Seeds and entropy coefficients to include in the plot
SEEDS_TO_PLOT = ["0","1","2"]
ENTROPY_COEFFS_TO_AVERAGE = ["1e-2","3e-2", "1e-3","3e-3","1e-1"]

# --- Smoothing Configuration ---
ROLLING_WINDOW_SIZE = 500 # <<< --- CHOOSE YOUR WINDOW SIZE HERE


plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif", 
    "font.serif": "Times",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 22,
})



# --- Helper function to get the mean run for a specific (coeff, type) across seeds ---
def get_mean_run_for_coeff_type(all_files, target_entropy_coeff, target_plot_type,
                                env_name, seeds_filter, window_size):
    """
    Loads, smooths, and averages runs across seeds for a specific entropy_coeff and plot_type.
    Returns: (mean_rewards_over_seeds, common_length, num_seed_runs) or (None, 0, 0).
    """
    df_list_smoothed = []

    for file_dir in all_files:
        try:
            parts = file_dir.split("_")
            seed = parts[2]
            entropy_coeff_from_file = parts[3]
            type_experiment_from_file = parts[4]
        except IndexError:
            continue

        if (env_name not in file_dir or
            seed not in seeds_filter or
            entropy_coeff_from_file != target_entropy_coeff or
            type_experiment_from_file != target_plot_type):
            continue

        file_path = os.path.join(logs_folder, file_dir, "progress.csv")
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
            if 'rollout/ep_rew_mean' not in df.columns or df.empty:
                continue

            rewards = df['rollout/ep_rew_mean']
            smoothed_rewards = rewards.rolling(window=window_size, min_periods=1).mean()
            df_list_smoothed.append(smoothed_rewards)
        except Exception:
            continue # Skip faulty files

    if not df_list_smoothed:
        return None, 0, 0

    min_len = min([len(s) for s in df_list_smoothed])
    aligned_dfs = [s[:min_len] for s in df_list_smoothed]

    if not aligned_dfs:
        return None, 0, 0

    data_array = np.array(aligned_dfs)
    mean_rewards_over_seeds = np.mean(data_array, axis=0) # Mean across seeds

    return mean_rewards_over_seeds, min_len, len(aligned_dfs)

# --- Script Start ---

# Read logs from the folder
try:
    all_log_dirs = os.listdir(logs_folder)
except FileNotFoundError:
    print(f"Error: Logs folder '{logs_folder}' not found.")
    exit()

if not ENTROPY_COEFFS_TO_AVERAGE:
    print("No entropy coefficients specified in ENTROPY_COEFFS_TO_AVERAGE.")
    exit()

print(f"Averaging across entropy coefficients: {ENTROPY_COEFFS_TO_AVERAGE}")
print(f"Averaging across seeds: {SEEDS_TO_PLOT}")

# Set up colors
colors = sns.color_palette("colorblind")
palette = {'complexity': colors[3], 'entropy': colors[0]}

# Create subplots - adjust layout based on number of environments
n_envs = len(environments)
n_cols = min(3, n_envs)  # Max 3 columns
n_rows = (n_envs + n_cols - 1) // n_cols  # Calculate rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
if n_envs == 1:
    axes = [axes]
elif n_rows == 1:
    axes = [axes]
else:
    axes = axes.flatten()

max_overall_common_length = 0 # To align the final two lines if needed

# Store handles and labels for the legend
legend_handles = []
legend_labels = []

# Iterate through all environments
for env_idx, environment in enumerate(environments):
    print(f"\n=== Processing Environment: {environment} ===")
    
    # Get the current axis for this environment
    ax = axes[env_idx]
    
    # Filter directories for this environment
    env_log_dirs = [d for d in all_log_dirs if environment in d and os.path.isdir(os.path.join(logs_folder, d))]
    
    if not env_log_dirs:
        print(f"No log directories found for environment '{environment}' in '{logs_folder}'. Skipping.")
        continue

    # Iterate through the two plot types: "complexity" and "entropy"
    for plot_type_to_aggregate in ["complexity", "entropy"]:
        print(f"\n--- Processing for Plot Type: {plot_type_to_aggregate} ---")

        # This list will store the mean reward curve (averaged over seeds) for each entropy coefficient
        mean_curves_per_coeff = []
        lengths_per_coeff = []
        num_coeffs_with_data = 0

        # For the current plot_type, get the mean run for each specified entropy coefficient
        for entropy_coeff in ENTROPY_COEFFS_TO_AVERAGE:
            mean_rewards_for_this_coeff, length, num_seed_runs = get_mean_run_for_coeff_type(
                env_log_dirs, entropy_coeff, plot_type_to_aggregate,
                environment, SEEDS_TO_PLOT, ROLLING_WINDOW_SIZE
            )

            if mean_rewards_for_this_coeff is not None and num_seed_runs > 0:
                mean_curves_per_coeff.append(mean_rewards_for_this_coeff)
                lengths_per_coeff.append(length)
                num_coeffs_with_data +=1
                print(f"  Got data for {plot_type_to_aggregate}, coeff {entropy_coeff} (avg over {num_seed_runs} seeds, len {length})")
            else:
                print(f"  No data or insufficient seed runs for {plot_type_to_aggregate}, coeff {entropy_coeff}")

        if not mean_curves_per_coeff:
            print(f"No data found for any entropy coefficient for plot type '{plot_type_to_aggregate}'. Skipping.")
            continue

        # Align all these mean_curves_per_coeff to their common minimum length
        common_min_len_for_plot_type = min(lengths_per_coeff)
        max_overall_common_length = max(max_overall_common_length, common_min_len_for_plot_type)

        aligned_mean_curves = [curve[:common_min_len_for_plot_type] for curve in mean_curves_per_coeff]

        if not aligned_mean_curves:
            print(f"No data after alignment for plot type '{plot_type_to_aggregate}'.")
            continue

        # Now, calculate the mean and std *across these aligned mean curves*
        # `data_array_of_means` will have shape (num_coeffs_with_data, common_min_len_for_plot_type)
        data_array_of_means = np.array(aligned_mean_curves)

        final_mean_across_coeffs = np.mean(data_array_of_means, axis=0)
        final_std_across_coeffs = np.std(data_array_of_means, axis=0)
        # Standard error of the mean (across coefficients)
        final_sem_across_coeffs = final_std_across_coeffs / np.sqrt(num_coeffs_with_data)

        # --- Plotting this aggregated line ---
        x_values = np.arange(common_min_len_for_plot_type)
        x_values = x_values * 256
        line, = ax.plot(x_values, final_mean_across_coeffs, label=f"{plot_type_to_aggregate}", 
                       color=palette[plot_type_to_aggregate])
        ax.fill_between(x_values,
                        final_mean_across_coeffs - final_sem_across_coeffs, # Using SEM
                        final_mean_across_coeffs + final_sem_across_coeffs,
                        alpha=0.2,
                        color=palette[plot_type_to_aggregate])
        
        # Store legend info from first subplot only
        if env_idx == 0:
            legend_handles.append(line)
            if plot_type_to_aggregate == "complexity":
                legend_labels.append(r"CRPPO (Avg $c_2$)")
            else:
                legend_labels.append(r"PPOwEnt (Avg $c_2$)")
        
        print(f"  Plotted aggregated line for {environment} {plot_type_to_aggregate}, final length {common_min_len_for_plot_type}")
    
    # Configure this subplot
    if ax.has_data():
        env_short = environment.split('_')[-1]
        ax.set_title(f"Number of Carts: {env_short}")
        ax.set_xlabel('Timesteps')
        
        # Only add y-label for leftmost subplots
        if env_idx % n_cols == 0:
            ax.set_ylabel('Mean Episode Reward')
        
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 510)
        ax.set_xlim(0, 15625* 256)
        ax.set_yticks(np.arange(0, 520, 100))
        ax.set_xticks(np.arange(0, 15625*256+1, 1e6))

# Hide empty subplots
for i in range(len(environments), len(axes)):
    axes[i].set_visible(False)

# Add a single legend at the top of the figure
if legend_handles:
    legend = fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2)

for line in legend.get_lines():
    line.set_linewidth(4) 
# --- Final Plot Configuration ---
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Make room for the legend
plot_filename = f"figures/meta_avg_plot_subplots_CARTerpillar.pdf"
plt.savefig(plot_filename, bbox_inches='tight')
print(f"\nPlot saved to {plot_filename}")