import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


def set_color_map(plt, palette: str = "colorblind") -> bool:
    """Sets the color cycle for the specified plot using a Seaborn palette."""
    try:
        colors = sns.color_palette(palette)
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
        return True
    except Exception as e:
        print(f"Error setting color map: {e}")
        return False


# Configure matplotlib settings
set_color_map(plt, "colorblind")
colors = sns.color_palette("colorblind")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": "Times",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 22,
    "figure.figsize": (21, 10)  # Taller to accommodate 2 rows
})

# Define color palette for entropy coefficients
COLOR_PALETTE = {
    '0': colors[-1],
    '1e-3': colors[2],
    '5e-3': colors[-2],
    '1e-2': colors[1],
    '5e-2': colors[7],
    '1e-1': colors[4]
}

# Shared configuration
LOGS_FOLDER = "logs"
ENTROPY_COEFFS_TO_PLOT = ["5e-2", "1e-2", "1e-3", "0", "1e-1", "5e-3"]
ENTROPY_COEFFS_TO_AVERAGE = ["1e-2", "1e-3", "5e-2", "1e-1", "5e-3"]
MAX_PLOT_POINTS = 5000

# Environment-specific configurations
ENV_CONFIGS = {
    "cartpole": {
        "environment": "CartPole-v1",
        "seeds": ["0", "1", "2", "3", "4"],
        "rolling_window": 10,
        "timestep_multiplier": 1,
        "xlim": (0, 1800),
        "ylim": (200, 510),
        "xticks": np.arange(0, 1801, 400),
        "progress_file": "progress.csv",
        "reward_column": "rollout/ep_rew_mean",
        "row_label": "CartPole-v1",
        "title": "CartPole",
    },
    "carracing": {
        "environment": "CarRacing-v2",
        "seeds": ["0", "1", "2", "3", "4"],
        "rolling_window": 100,
        "timestep_multiplier": 1,
        "xlim": (0, 4800),
        "ylim": (300, 1000),
        "xticks": np.arange(0, 4801, 1000),
        "progress_file": "progress.csv",
        "reward_column": "rollout/ep_rew_mean",
        "row_label": "CarRacing-v2",
        "title": "CarRacing",
    },
}


def load_and_process_data(logs_folder, environment, plot_type, entropy_coefficient,
                          seeds_to_plot, window_size, progress_file, reward_column, max_points=None):
    """Load and process data for a specific entropy coefficient and plot type."""
    files = [f for f in os.listdir(logs_folder)
             if environment in f and os.path.isdir(os.path.join(logs_folder, f))]

    df_list_smoothed = []

    for file in files:
        try:
            parts = file.split("_")
            seed = parts[1]
            entropy_coeff = parts[2]
            type_experiment = parts[3]
        except IndexError:
            continue

        if (entropy_coeff != entropy_coefficient or
                type_experiment != plot_type or
                seed not in seeds_to_plot):
            continue

        file_path = os.path.join(logs_folder, file, progress_file)
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
            if reward_column not in df.columns:
                continue

            rewards = df[reward_column]
            smoothed_rewards = rewards.rolling(window=window_size, min_periods=1).mean()
            df_list_smoothed.append(smoothed_rewards)

        except Exception:
            continue

    if not df_list_smoothed:
        return None, None, None

    # Align lengths
    min_len = min(len(df) for df in df_list_smoothed)
    aligned_dfs = [df[:min_len] for df in df_list_smoothed]

    # Calculate statistics
    data_array = np.array(aligned_dfs)
    mean_smoothed = np.mean(data_array, axis=0)
    std_smoothed = np.std(data_array, axis=0)
    sem_smoothed = std_smoothed / np.sqrt(len(aligned_dfs))

    x_values = np.arange(min_len)

    # Downsample if data exceeds max_points threshold
    if max_points is not None and len(x_values) > max_points:
        step = len(x_values) // max_points
        if step > 1:
            indices = np.arange(0, len(x_values), step)
            x_values = x_values[indices]
            mean_smoothed = mean_smoothed[indices]
            sem_smoothed = sem_smoothed[indices]

    return x_values, mean_smoothed, sem_smoothed


def get_mean_run_for_coeff_type(all_files, target_entropy_coeff, target_plot_type,
                                 environment, seeds_to_plot, window_size, progress_file, reward_column):
    """Get mean run for a specific entropy coefficient and plot type across seeds."""
    df_list_smoothed = []

    for file_dir in all_files:
        try:
            parts = file_dir.split("_")
            seed = parts[1]
            entropy_coeff_from_file = parts[2]
            type_experiment_from_file = parts[3]
        except IndexError:
            continue

        if (environment not in file_dir or
                seed not in seeds_to_plot or
                entropy_coeff_from_file != target_entropy_coeff or
                type_experiment_from_file != target_plot_type):
            continue

        file_path = os.path.join(LOGS_FOLDER, file_dir, progress_file)
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
            if reward_column not in df.columns or df.empty:
                continue

            rewards = df[reward_column]
            smoothed_rewards = rewards.rolling(window=window_size, min_periods=1).mean()
            df_list_smoothed.append(smoothed_rewards)
        except Exception:
            continue

    if not df_list_smoothed:
        return None, 0, 0

    min_len = min([len(s) for s in df_list_smoothed])
    aligned_dfs = [s[:min_len] for s in df_list_smoothed]

    if not aligned_dfs:
        return None, 0, 0

    data_array = np.array(aligned_dfs)
    mean_rewards_over_seeds = np.mean(data_array, axis=0)

    return mean_rewards_over_seeds, min_len, len(aligned_dfs)


def plot_row(fig, axes, env_key, config, is_top_row):
    """Plot a single row (3 subplots) for one environment."""
    ax1, ax2, ax3 = axes

    environment = config["environment"]
    seeds = config["seeds"]
    window_size = config["rolling_window"]
    multiplier = config["timestep_multiplier"]
    progress_file = config["progress_file"]
    reward_column = config["reward_column"]

    # Check if logs folder exists
    try:
        files = os.listdir(LOGS_FOLDER)
    except FileNotFoundError:
        print(f"Error: Logs folder '{LOGS_FOLDER}' not found.")
        return [], [], [], []

    # Filter relevant files
    files = [f for f in files if environment in f and os.path.isdir(os.path.join(LOGS_FOLDER, f))]

    if not files:
        print(f"No log directories found for environment '{environment}' in '{LOGS_FOLDER}'.")
        return [], [], [], []

    # First two plots: individual entropy coefficients
    plot_configs = [
        ("complexity", ax1, "CR-PPO"),
        ("entropy", ax2, "PPOwEnt")
    ]

    # Store all lines for the combined legend
    all_lines = []
    all_labels = []

    for plot_type, ax, title in plot_configs:
        print(f"Processing {env_key} {plot_type} plot...")

        # Get available entropy coefficients for this plot type
        available_coeffs = []
        for file in files:
            try:
                parts = file.split("_")
                seed = parts[1]
                entropy_coeff = parts[2]
                type_experiment = parts[3]

                if (seed in seeds and
                        entropy_coeff in ENTROPY_COEFFS_TO_PLOT and
                        type_experiment == plot_type and
                        entropy_coeff not in available_coeffs):
                    available_coeffs.append(entropy_coeff)
            except IndexError:
                continue

        # Sort coefficients numerically
        try:
            available_coeffs.sort(key=float)
        except ValueError:
            available_coeffs.sort()

        if not available_coeffs:
            print(f"No data found for {plot_type}")
            continue

        # Plot each entropy coefficient
        for entropy_coeff in available_coeffs:
            x_vals, mean_vals, sem_vals = load_and_process_data(
                LOGS_FOLDER, environment, plot_type, entropy_coeff,
                seeds, window_size, progress_file, reward_column, MAX_PLOT_POINTS
            )

            if x_vals is not None:
                color = COLOR_PALETTE.get(entropy_coeff, colors[0])
                x_vals = x_vals * multiplier

                line, = ax.plot(x_vals, mean_vals, label=f"{entropy_coeff}", color=color)
                ax.fill_between(x_vals,
                                mean_vals - sem_vals,
                                mean_vals + sem_vals,
                                alpha=0.2, color=color)

                # Store line info for combined legend (only from first plot)
                if plot_type == "complexity":
                    all_lines.append(line)
                    all_labels.append(f"{entropy_coeff}")

        if is_top_row:
            ax.set_title(title)
        ax.set_xlabel('Timesteps')
        if ax == ax1:
            ax.set_ylabel('Mean Episode Reward')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(*config["xlim"])
        ax.set_ylim(*config["ylim"])
        ax.set_xticks(config["xticks"])

        # Remove Y-axis labels for subplots 2 and 3
        if ax != ax1:
            ax.set_yticklabels([])

    # Third plot: comparison (averaged across entropy coefficients)
    print(f"Processing {env_key} comparison plot...")

    # Colors for comparison plot
    comparison_palette = {'complexity': colors[3], 'entropy': colors[0]}
    comparison_lines = []
    comparison_labels = []

    for plot_type_to_aggregate in ["complexity", "entropy"]:
        print(f"Processing for Plot Type: {plot_type_to_aggregate}")

        mean_curves_per_coeff = []
        lengths_per_coeff = []
        num_coeffs_with_data = 0

        # Get mean run for each entropy coefficient
        for entropy_coeff in ENTROPY_COEFFS_TO_AVERAGE:
            mean_rewards_for_this_coeff, length, num_seed_runs = get_mean_run_for_coeff_type(
                files, entropy_coeff, plot_type_to_aggregate,
                environment, seeds, window_size, progress_file, reward_column
            )

            if mean_rewards_for_this_coeff is not None and num_seed_runs > 0:
                mean_curves_per_coeff.append(mean_rewards_for_this_coeff)
                lengths_per_coeff.append(length)
                num_coeffs_with_data += 1
                print(f"  Got data for {plot_type_to_aggregate}, coeff {entropy_coeff} "
                      f"(avg over {num_seed_runs} seeds, len {length})")

        if not mean_curves_per_coeff:
            print(f"No data found for any entropy coefficient for plot type '{plot_type_to_aggregate}'.")
            continue

        # Align curves
        common_min_len_for_plot_type = min(lengths_per_coeff)
        aligned_mean_curves = [curve[:common_min_len_for_plot_type] for curve in mean_curves_per_coeff]

        if not aligned_mean_curves:
            print(f"No data after alignment for plot type '{plot_type_to_aggregate}'.")
            continue

        # Calculate mean and SEM across coefficients
        data_array_of_means = np.array(aligned_mean_curves)
        final_mean_across_coeffs = np.mean(data_array_of_means, axis=0)
        final_std_across_coeffs = np.std(data_array_of_means, axis=0)
        final_sem_across_coeffs = final_std_across_coeffs / np.sqrt(num_coeffs_with_data)

        # Plot aggregated line
        x_values = np.arange(common_min_len_for_plot_type) * multiplier
        line, = ax3.plot(x_values, final_mean_across_coeffs,
                         label=f"{plot_type_to_aggregate}",
                         color=comparison_palette[plot_type_to_aggregate])
        ax3.fill_between(x_values,
                         final_mean_across_coeffs - final_sem_across_coeffs,
                         final_mean_across_coeffs + final_sem_across_coeffs,
                         alpha=0.2,
                         color=comparison_palette[plot_type_to_aggregate])

        comparison_lines.append(line)
        if plot_type_to_aggregate == "complexity":
            comparison_labels.append(r"CR-PPO (Avg)")
        else:
            comparison_labels.append(r"PPOwEnt (Avg)")

    # Configure third subplot
    if is_top_row:
        ax3.set_title("Comparison")
    ax3.set_xlabel('Timesteps')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xlim(*config["xlim"])
    ax3.set_ylim(*config["ylim"])
    ax3.set_xticks(config["xticks"])
    ax3.set_yticklabels([])  # Remove Y-axis labels

    # Add environment name as rotated text on the right side of the row
    ax3.text(1.05, 0.5, config["title"], rotation=-90,
             ha='left', va='center', transform=ax3.transAxes)

    return all_lines, all_labels, comparison_lines, comparison_labels


def create_six_subplot_figure():
    """Create a 2x3 subplot figure combining CartPole (top) and CarRacing (bottom)."""
    fig, axes = plt.subplots(2, 3)

    # Row 1: CartPole
    all_lines_top, all_labels_top, comp_lines_top, comp_labels_top = plot_row(
        fig, axes[0], "cartpole", ENV_CONFIGS["cartpole"], is_top_row=True
    )

    # Row 2: CarRacing
    all_lines_bot, all_labels_bot, comp_lines_bot, comp_labels_bot = plot_row(
        fig, axes[1], "carracing", ENV_CONFIGS["carracing"], is_top_row=False
    )

    # Use the lines from whichever row has data for the shared legend
    all_lines = all_lines_top if all_lines_top else all_lines_bot
    all_labels = all_labels_top if all_labels_top else all_labels_bot
    comparison_lines = comp_lines_top if comp_lines_top else comp_lines_bot
    comparison_labels = comp_labels_top if comp_labels_top else comp_labels_bot

    # Create legends
    if all_lines:
        legend1 = fig.legend(all_lines, all_labels, loc='lower center',
                             bbox_to_anchor=(0.375, -0.05), ncol=len(all_labels), frameon=False)
        for line in legend1.get_lines():
            line.set_linewidth(4)

    if comparison_lines:
        legend2 = fig.legend(comparison_lines, comparison_labels, loc='lower center',
                             bbox_to_anchor=(0.833, -0.05), title="", ncol=len(comparison_labels), frameon=False)
        for line in legend2.get_lines():
            line.set_linewidth(4)

    # Add text to the left of legend1
    fig.text(0.06, 0.01, r"Coeff ($c_{reg}$)", fontsize=22, ha='center', va='center')

    plt.tight_layout()

    # Save figure
    plot_filename = "figures/six_subplot_cartpole_carracing.pdf"
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Six subplot plot saved to {plot_filename}")


if __name__ == "__main__":
    create_six_subplot_figure()
    print("Finished generating plots.")
