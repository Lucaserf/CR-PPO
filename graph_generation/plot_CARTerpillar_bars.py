import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import seaborn as sns

def set_color_map(plt, palette: str = "colorblind") -> bool:
    """
    Sets the color cycle for the specified plot using a Seaborn palette.
    """
    try:
        colors = sns.color_palette(palette)
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
        return True
    except Exception as e:
        print(f"Error setting color map: {e}")
        return False

set_color_map(plt, "colorblind")

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": "Times",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 22,
    }
)
use_pgf = True
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = (8,5)

colors = sns.color_palette("colorblind")

# --- Configuration ---
logs_folder = "logs"

environments = ["ComplexCartPoleEnv9.81_6","ComplexCartPoleEnv9.81_7","ComplexCartPoleEnv9.81_8", "ComplexCartPoleEnv9.81_9", "ComplexCartPoleEnv9.81_10", "ComplexCartPoleEnv9.81_11"]
SEEDS_TO_PLOT = ["0","1","2"]
ENTROPY_COEFFS_TO_AVERAGE = ["1e-2","3e-2", "1e-3","3e-3","1e-1"]
ROLLING_WINDOW_SIZE = 500

# --- Helper function to get final performance statistics ---
def get_final_performance_stats(all_files, target_entropy_coeff, target_plot_type,
                               env_name, seeds_filter, window_size, final_fraction=0.1):
    """
    Get final performance statistics (mean of last 10% of episodes) for a specific configuration.
    Returns: (final_mean, final_std, final_sem, num_runs) or (None, None, None, 0)
    """
    final_values = []

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
            
            # Take the mean of the final 10% of episodes
            final_portion_size = max(1, int(len(smoothed_rewards) * final_fraction))
            final_mean_reward = smoothed_rewards[-final_portion_size:].mean()
            final_values.append(final_mean_reward)
            
        except Exception:
            continue

    if not final_values:
        return None, None, None, 0

    final_values = np.array(final_values)
    mean_val = np.mean(final_values)
    std_val = np.std(final_values)
    sem_val = std_val / np.sqrt(len(final_values))
    
    return mean_val, std_val, sem_val, len(final_values)

# --- Get aggregated performance for complexity and entropy coefficients ---
def get_aggregated_performance_stats(all_files, entropy_coeffs_list, target_plot_type,
                                   env_name, seeds_filter, window_size, final_fraction=0.1):
    """
    Get aggregated performance statistics across multiple entropy coefficients.
    """
    all_final_means = []
    
    for entropy_coeff in entropy_coeffs_list:
        final_mean, _, _, num_runs = get_final_performance_stats(
            all_files, entropy_coeff, target_plot_type,
            env_name, seeds_filter, window_size, final_fraction
        )
        
        if final_mean is not None and num_runs > 0:
            all_final_means.append(final_mean)
    
    if not all_final_means:
        return None, None, None, 0
    
    all_final_means = np.array(all_final_means)
    mean_val = np.mean(all_final_means)
    std_val = np.std(all_final_means)
    sem_val = std_val / np.sqrt(len(all_final_means))
    
    return mean_val, std_val, sem_val, len(all_final_means)

# --- Read logs ---
try:
    all_log_dirs = os.listdir(logs_folder)
except FileNotFoundError:
    print(f"Error: Logs folder '{logs_folder}' not found.")
    exit()

print("Creating comparison table across all environments")
print("="*80)

# Initialize results dictionary with environments as keys
results = {}
for env in environments:
    results[env] = {}

# Process each environment
for environment in environments:
    print(f"\nProcessing environment: {environment}")
    
    # Filter log directories for this environment
    env_log_dirs = [d for d in all_log_dirs if environment in d and os.path.isdir(os.path.join(logs_folder, d))]
    
    if not env_log_dirs:
        print(f"No log directories found for environment '{environment}'.")
        # Set all values to None for this environment
        results[environment]["Baseline"] = None
        results[environment]["Mean complexity coeffs"] = None
        results[environment]["Mean entropy coeffs"] = None
        continue
    
    # --- Get statistics for coefficient 0 baseline (use complexity data) ---
    baseline_mean, baseline_std, baseline_sem, baseline_runs = get_final_performance_stats(
        env_log_dirs, "0", "complexity", environment, SEEDS_TO_PLOT, ROLLING_WINDOW_SIZE
    )
    
    if baseline_mean is not None:
        results[environment]["Baseline"] = f"{baseline_mean:.2f} ± {baseline_sem:.2f}"
    else:
        results[environment]["Baseline"] = "N/A"
    
    # --- Get aggregated statistics for complexity and entropy mean coefficients ---
    for plot_type in ["complexity", "entropy"]:
        agg_mean, agg_std, agg_sem, agg_n = get_aggregated_performance_stats(
            env_log_dirs, ENTROPY_COEFFS_TO_AVERAGE, plot_type,
            environment, SEEDS_TO_PLOT, ROLLING_WINDOW_SIZE
        )
        
        if agg_mean is not None:
            results[environment][f"Mean {plot_type} coeffs"] = f"{agg_mean:.2f} ± {agg_sem:.2f}"
        else:
            results[environment][f"Mean {plot_type} coeffs"] = "N/A"

# --- Create and display table ---
# Define the row labels (conditions)
conditions = ["Baseline", "Mean complexity coeffs", "Mean entropy coeffs"]

# Create table data
table_data = []
for condition in conditions:
    row = {"Condition": condition}
    for env in environments:
        row[env] = results[env].get(condition, "N/A")
    table_data.append(row)

# Create DataFrame and display
comparison_df = pd.DataFrame(table_data)
print("\nFinal Performance Comparison Table (Mean ± Std Error):")
print("="*80)
print(comparison_df.to_string(index=False))

# --- Create a formatted table plot ---
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=comparison_df.values,
                colLabels=comparison_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.5)

# Style header row
for i in range(len(comparison_df.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows with alternating colors
for i in range(1, len(comparison_df) + 1):
    for j in range(len(comparison_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')
        
        # Make condition column bold
        if j == 0:
            table[(i, j)].set_text_props(weight='bold')

plt.title('Performance Comparison Across Environments (Mean ± SEM)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

# Save table
table_filename = f"figures/performance_table_all_environments.pdf"
plt.savefig(table_filename, bbox_inches='tight', dpi=300)
print(f"\nTable saved to {table_filename}")

# --- Save table as CSV ---
csv_filename = f"figures/performance_table_all_environments.csv"
comparison_df.to_csv(csv_filename, index=False)
print(f"Table data saved to {csv_filename}")

print("\nTable generation complete!")

# Extract numerical values from the results for plotting
plot_data = {'environments': [], 'baseline': [], 'baseline_sem': [], 
             'complexity': [], 'complexity_sem': [], 'entropy': [], 'entropy_sem': []}

for env in environments:
    plot_data['environments'].append(env.replace('ComplexCartPoleEnv9.81_', ''))
    
    # Parse baseline values
    baseline_str = results[env]["Baseline"]
    if baseline_str != "N/A" and baseline_str is not None:
        baseline_val, baseline_sem_val = baseline_str.split(' ± ')
        plot_data['baseline'].append(float(baseline_val))
        plot_data['baseline_sem'].append(float(baseline_sem_val))
    else:
        plot_data['baseline'].append(0)
        plot_data['baseline_sem'].append(0)
    
    # Parse complexity values
    complexity_str = results[env]["Mean complexity coeffs"]
    if complexity_str != "N/A" and complexity_str is not None:
        complexity_val, complexity_sem_val = complexity_str.split(' ± ')
        plot_data['complexity'].append(float(complexity_val))
        plot_data['complexity_sem'].append(float(complexity_sem_val))
    else:
        plot_data['complexity'].append(0)
        plot_data['complexity_sem'].append(0)
    
    # Parse entropy values
    entropy_str = results[env]["Mean entropy coeffs"]
    if entropy_str != "N/A" and entropy_str is not None:
        entropy_val, entropy_sem_val = entropy_str.split(' ± ')
        plot_data['entropy'].append(float(entropy_val))
        plot_data['entropy_sem'].append(float(entropy_sem_val))
    else:
        plot_data['entropy'].append(0)
        plot_data['entropy_sem'].append(0)

# Create the plot
fig, ax = plt.subplots(figsize=(14, 5.5))

x = np.arange(len(plot_data['environments']))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, plot_data['baseline'], width, 
               yerr=plot_data['baseline_sem'], capsize=5,
               label='PPO', alpha=0.8, color=colors[-1])

bars2 = ax.bar(x , plot_data['entropy'], width,
               yerr=plot_data['entropy_sem'], capsize=5,
               label='PPOwEnt', alpha=0.8, color=colors[0])


bars3 = ax.bar(x + width, plot_data['complexity'], width,
               yerr=plot_data['complexity_sem'], capsize=5,
               label='CDPO', alpha=0.8, color=colors[3])


# Customize the plot
ax.set_xlabel('Number of Carts')
ax.set_ylabel('Mean Episode Reward')
ax.set_xticks(x)
ax.set_xticklabels(plot_data['environments'])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
# def add_value_labels(bars, values, sems):
#     for bar, val, sem in zip(bars, values, sems):
#         if val > 0:  # Only add label if value is not zero (i.e., data exists)
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height + sem + 5,
#                    f'{val:.1f}', ha='center', va='bottom', fontsize=22)

# add_value_labels(bars1, plot_data['baseline'], plot_data['baseline_sem'])
# add_value_labels(bars2, plot_data['complexity'], plot_data['complexity_sem'])
# add_value_labels(bars3, plot_data['entropy'], plot_data['entropy_sem'])

# Add legend outside of the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=22)
# ax.legend()

# Set y-axis limit to accommodate error bars and labels
max_val = max(max(plot_data['baseline']), max(plot_data['complexity']), max(plot_data['entropy']))
max_sem = max(max(plot_data['baseline_sem']), max(plot_data['complexity_sem']), max(plot_data['entropy_sem']))
ax.set_ylim(0, 525)  # Add some padding for labels
#set yticks
ax.set_yticks(np.arange(0, 550, 100))


plt.tight_layout()

# Save the plot
plot_comparison_filename = "figures/performance_comparison_plot_all_environments.pdf"
plt.savefig(plot_comparison_filename, bbox_inches='tight', dpi=300)
print(f"\nComparison plot saved to {plot_comparison_filename}")


print("\nComparison plot generation complete!")