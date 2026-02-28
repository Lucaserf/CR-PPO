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
    "figure.figsize": (14, 5)  # Two subplots side by side
})

# Define color palette for entropy coefficients
COLOR_PALETTE = {
    '0': colors[-1],
    '1e-3': colors[2],
    '3e-3': colors[-2],
    '1e-2': colors[1],
    '3e-2': colors[7],
    '1e-1': colors[4]
}

# Configuration
LOGS_FOLDER = "logs"
ENVIRONMENTS = [
    "ComplexCartPoleEnv9.81_6",
    "ComplexCartPoleEnv9.81_7",
    "ComplexCartPoleEnv9.81_8",
    "ComplexCartPoleEnv9.81_9",
    "ComplexCartPoleEnv9.81_10",
    "ComplexCartPoleEnv9.81_11",
]

SEEDS_TO_PLOT = ["0", "1", "2"]
ENTROPY_COEFFS_TO_PLOT = ["1e-2", "3e-2", "1e-3", "3e-3", "1e-1", "0"]
ROLLING_WINDOW_SIZE = 500
MAX_PLOT_POINTS = 5000

# X-axis multiplier (each timestep corresponds to 256 environment steps)
X_MULTIPLIER = 256


def load_and_process_data(logs_folder, environment, plot_type, entropy_coefficient,
                          seeds_to_plot, window_size, max_points=None):
    """Load and process data for a specific entropy coefficient and plot type.

    CARTerpillar log directory naming convention:
        <env>_<something>_<seed>_<entropy_coeff>_<type>
    So after splitting on '_', the relevant indices are:
        seed -> parts[2], entropy_coeff -> parts[3], type -> parts[4]
    """
    files = [f for f in os.listdir(logs_folder)
             if environment in f and os.path.isdir(os.path.join(logs_folder, f))]

    df_list_smoothed = []

    for file in files:
        try:
            parts = file.split("_")
            seed = parts[2]
            entropy_coeff = parts[3]
            type_experiment = parts[4]
        except IndexError:
            continue

        if (entropy_coeff != entropy_coefficient or
                type_experiment != plot_type or
                seed not in seeds_to_plot):
            continue

        file_path = os.path.join(logs_folder, file, "progress.csv")
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
            if 'rollout/ep_rew_mean' not in df.columns:
                continue

            rewards = df['rollout/ep_rew_mean']
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


def create_two_subplot_figure(environment):
    """Create a two-subplot figure (CR-PPO and PPOwEnt) for a single CARTerpillar environment."""
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Check if logs folder exists
    try:
        files = os.listdir(LOGS_FOLDER)
    except FileNotFoundError:
        print(f"Error: Logs folder '{LOGS_FOLDER}' not found.")
        return

    # Filter relevant files for this environment
    files = [f for f in files if environment in f and os.path.isdir(os.path.join(LOGS_FOLDER, f))]

    if not files:
        print(f"No log directories found for environment '{environment}' in '{LOGS_FOLDER}'.")
        return

    # Two plots: individual entropy coefficients for CR-PPO and PPOwEnt
    plot_configs = [
        ("complexity", ax1, "CR-PPO"),
        ("entropy", ax2, "PPO")
    ]

    # Store all lines for the combined legend
    all_lines = []
    all_labels = []

    for plot_type, ax, title in plot_configs:
        print(f"  Processing {plot_type} plot...")

        # Get available entropy coefficients for this plot type
        available_coeffs = []
        for file in files:
            try:
                parts = file.split("_")
                seed = parts[2]
                entropy_coeff = parts[3]
                type_experiment = parts[4]

                if (seed in SEEDS_TO_PLOT and
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
            print(f"  No data found for {plot_type}")
            continue

        # Plot each entropy coefficient
        for entropy_coeff in available_coeffs:
            x_vals, mean_vals, sem_vals = load_and_process_data(
                LOGS_FOLDER, environment, plot_type, entropy_coeff,
                SEEDS_TO_PLOT, ROLLING_WINDOW_SIZE, MAX_PLOT_POINTS
            )

            if x_vals is not None:
                color = COLOR_PALETTE.get(entropy_coeff, colors[0])
                x_vals = x_vals * X_MULTIPLIER

                line, = ax.plot(x_vals, mean_vals, label=f"{entropy_coeff}", color=color)
                ax.fill_between(x_vals,
                                mean_vals - sem_vals,
                                mean_vals + sem_vals,
                                alpha=0.2, color=color)

                # Store line info for combined legend (only from first plot)
                if plot_type == "complexity":
                    all_lines.append(line)
                    all_labels.append(f"{entropy_coeff}")

        ax.set_title(title)
        ax.set_xlabel('Timesteps')
        if ax == ax1:
            ax.set_ylabel('Mean Episode Reward')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 510)
        ax.set_xlim(0, 15625 * X_MULTIPLIER)
        ax.set_yticks(np.arange(0, 520, 100))
        ax.set_xticks(np.arange(0, 15625 * X_MULTIPLIER + 1, 1e6))

        # Remove Y-axis labels for the second subplot
        if ax != ax1:
            ax.set_yticklabels([])

    # Create legend to the right of the plots
    if all_lines:
        legend = fig.legend(all_lines, all_labels, loc='center left',
                            bbox_to_anchor=(0.98, 0.55), ncol=1, frameon=True,
                            title=r"Coeff ($c_{reg}$)")

        for line in legend.get_lines():
            line.set_linewidth(4)

    plt.tight_layout()

    # Save figure
    os.makedirs("figures", exist_ok=True)
    plot_filename = f"figures/combined_plot_{environment}.pdf"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved to {plot_filename}")


if __name__ == "__main__":
    for env in ENVIRONMENTS:
        print(f"\n=== Processing Environment: {env} ===")
        create_two_subplot_figure(env)
    print("\nFinished generating all plots.")
