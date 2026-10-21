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
LOGS_FOLDER = "D:\\CR-PPO\\logs"
ENTROPY_COEFFS_TO_PLOT = ["5e-2", "1e-2", "1e-3", "0", "1e-1", "5e-3"]
ENTROPY_COEFFS_TO_AVERAGE = ["1e-2", "1e-3", "5e-2", "1e-1", "5e-3"]

# Environment-specific configurations (aligned by actual environment timesteps)
ENV_CONFIGS = {
    "cartpole": {
        "environment": "CartPole-v1",
        "seeds": ["0", "1", "2", "3", "4"],
        "max_timesteps": 500000,
        "num_grid_points": 1000,
        "smoothing_timesteps": 2500,
        "xlim": (0, 500000),
        "ylim": (200, 510),
        "xticks": np.arange(0, 500001, 100000),
        "progress_file": "progress.csv",
        "reward_column": "rollout/ep_rew_mean",
        "row_label": "CartPole-v1",
        "title": "CartPole",
    },
    "carracing": {
        "environment": "CarRacing-v2",
        "seeds": ["0", "1", "2", "3", "4"],
        "max_timesteps": 10000000,
        "num_grid_points": 2000,
        "smoothing_timesteps": 200000,
        "xlim": (0, 4000000),
        "ylim": (300, 1000),
        "xticks": np.arange(0, 4000001, 2000000),
        "progress_file": "progress.csv",
        "reward_column": "rollout/ep_rew_mean",
        "row_label": "CarRacing-v2",
        "title": "CarRacing",
    },
}


def load_run_data(file_path, grid, smoothing_timesteps, reward_column="rollout/ep_rew_mean"):
    """Load a single run CSV, interpolate onto a uniform timestep grid, and apply rolling smoothing."""
    try:
        df = pd.read_csv(file_path)
        if "time/total_timesteps" not in df.columns or reward_column not in df.columns:
            return None

        sub = df[["time/total_timesteps", reward_column]].dropna()
        if sub.empty:
            return None

        t = sub["time/total_timesteps"].values.astype(float)
        r = sub[reward_column].values.astype(float)

        # Ensure monotonically increasing timesteps
        idx = np.argsort(t)
        t, r = t[idx], r[idx]
        t, unique_idx = np.unique(t, return_index=True)
        r = r[unique_idx]

        # Interpolate onto shared uniform grid
        interp_r = np.interp(grid, t, r).copy()
        # Mark timesteps beyond this run's actual final timestep as NaN
        interp_r[grid > t[-1]] = np.nan

        # Apply rolling window smoothing on the uniform timestep grid
        if smoothing_timesteps and len(grid) > 1:
            grid_step = grid[1] - grid[0]
            pts = max(1, int(round(smoothing_timesteps / grid_step)))
            interp_r = pd.Series(interp_r).rolling(window=pts, min_periods=1).mean().to_numpy().copy()
            # Preserve NaNs after smoothing
            interp_r[grid > t[-1]] = np.nan

        return interp_r
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_and_process_data(logs_folder, config, plot_type, entropy_coeff):
    """Load and process data across all seeds for a specific entropy coefficient and plot type."""
    environment = config["environment"]
    seeds = config["seeds"]
    grid = np.linspace(0, config["max_timesteps"], config["num_grid_points"])
    smoothing_timesteps = config.get("smoothing_timesteps", 0)
    progress_file = config["progress_file"]
    reward_column = config["reward_column"]

    runs = []
    for seed in seeds:
        folder = f"{environment}_{seed}_{entropy_coeff}_{plot_type}"
        file_path = os.path.join(logs_folder, folder, progress_file)
        if not os.path.exists(file_path):
            continue
        data = load_run_data(file_path, grid, smoothing_timesteps, reward_column)
        if data is not None:
            runs.append(data)

    if not runs:
        return None, None, None

    arr = np.array(runs)  # shape: (n_seeds, num_grid_points)
    valid_counts = np.sum(~np.isnan(arr), axis=0)
    mask = valid_counts > 0

    mean_vals = np.full(len(grid), np.nan)
    sem_vals = np.full(len(grid), np.nan)

    mean_vals[mask] = np.nanmean(arr[:, mask], axis=0)
    std_vals = np.nanstd(arr[:, mask], axis=0)
    sem_vals[mask] = std_vals / np.sqrt(valid_counts[mask])

    return grid, mean_vals, sem_vals


def get_mean_run_for_coeff_type(logs_folder, config, target_entropy_coeff, target_plot_type):
    """Get mean reward curve across seeds for a given coefficient and plot type on the shared grid."""
    _, mean_vals, _ = load_and_process_data(logs_folder, config, target_plot_type, target_entropy_coeff)
    return mean_vals


def plot_row(fig, axes, env_key, config, is_top_row):
    """Plot a single row (3 subplots) for one environment."""
    ax1, ax2, ax3 = axes

    environment = config["environment"]
    seeds = config["seeds"]
    grid = np.linspace(0, config["max_timesteps"], config["num_grid_points"])

    try:
        files = os.listdir(LOGS_FOLDER)
    except FileNotFoundError:
        print(f"Error: Logs folder '{LOGS_FOLDER}' not found.")
        return [], [], [], []

    files = [f for f in files if environment in f and os.path.isdir(os.path.join(LOGS_FOLDER, f))]
    if not files:
        print(f"No log directories found for environment '{environment}' in '{LOGS_FOLDER}'.")
        return [], [], [], []

    # First two plots: individual entropy coefficients
    plot_configs = [
        ("complexity", ax1, "CR-PPO"),
        ("entropy", ax2, "PPOwEnt")
    ]

    all_lines = []
    all_labels = []

    for plot_type, ax, title in plot_configs:
        print(f"Processing {env_key} {plot_type} plot...")

        available_coeffs = []
        for file in files:
            try:
                parts = file.split("_")
                seed, entropy_coeff, type_experiment = parts[1], parts[2], parts[3]
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
                LOGS_FOLDER, config, plot_type, entropy_coeff
            )

            if x_vals is not None:
                color = COLOR_PALETTE.get(entropy_coeff, colors[0])
                valid_mask = ~np.isnan(mean_vals)
                x_plot = x_vals[valid_mask]
                mean_plot = mean_vals[valid_mask]
                sem_plot = sem_vals[valid_mask]

                line, = ax.plot(x_plot, mean_plot, label=f"{entropy_coeff}", color=color)
                ax.fill_between(x_plot,
                                mean_plot - sem_plot,
                                mean_plot + sem_plot,
                                alpha=0.2, color=color)

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
    comparison_palette = {'complexity': colors[3], 'entropy': colors[0]}
    comparison_lines = []
    comparison_labels = []

    for plot_type_to_aggregate in ["complexity", "entropy"]:
        print(f"Processing for Plot Type: {plot_type_to_aggregate}")
        mean_curves_per_coeff = []

        for entropy_coeff in ENTROPY_COEFFS_TO_AVERAGE:
            mean_rewards = get_mean_run_for_coeff_type(
                LOGS_FOLDER, config, entropy_coeff, plot_type_to_aggregate
            )
            if mean_rewards is not None:
                mean_curves_per_coeff.append(mean_rewards)
                print(f"  Got data for {plot_type_to_aggregate}, coeff {entropy_coeff}")

        if not mean_curves_per_coeff:
            print(f"No data found for any entropy coefficient for plot type '{plot_type_to_aggregate}'.")
            continue

        arr_means = np.array(mean_curves_per_coeff)  # shape: (n_coeffs, num_grid_points)
        valid_counts = np.sum(~np.isnan(arr_means), axis=0)
        mask = valid_counts > 0

        final_mean = np.full(len(grid), np.nan)
        final_sem = np.full(len(grid), np.nan)

        final_mean[mask] = np.nanmean(arr_means[:, mask], axis=0)
        final_std = np.nanstd(arr_means[:, mask], axis=0)
        final_sem[mask] = final_std / np.sqrt(valid_counts[mask])

        x_plot = grid[mask]
        y_plot = final_mean[mask]
        sem_plot = final_sem[mask]

        line, = ax3.plot(x_plot, y_plot,
                         label=f"{plot_type_to_aggregate}",
                         color=comparison_palette[plot_type_to_aggregate])
        ax3.fill_between(x_plot,
                         y_plot - sem_plot,
                         y_plot + sem_plot,
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
    ax3.set_yticklabels([])

    # Add environment name as rotated text on the right side of the row
    ax3.text(1.05, 0.5, config["title"], rotation=-90,
             ha='left', va='center', transform=ax3.transAxes)

    # Force exponential/scientific notation on the x-axis ticks
    for ax in axes:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    return all_lines, all_labels, comparison_lines, comparison_labels


def create_six_subplot_figure(output_path="figures/six_subplot_cartpole_carracing_fixed.pdf"):
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Six subplot plot saved to {output_path}")

    # Also save standard filename for convenience
    standard_path = "figures/six_subplot_cartpole_carracing.pdf"
    if output_path != standard_path:
        plt.savefig(standard_path, bbox_inches='tight')
        print(f"Six subplot plot also saved to {standard_path}")


if __name__ == "__main__":
    create_six_subplot_figure()
    print("Finished generating plots.")
