import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

# === Smoothing Function ===
def smooth_data(data, window_size):
    """Smooth data using a moving average with padding at the edges."""
    if len(data) < window_size:
        return data  # If the data is too short, return it unchanged
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    smoothed_data = np.convolve(
        padded_data, 
        np.ones(window_size) / window_size, 
        mode='valid'
    )
    return smoothed_data[:len(data)]


# === Reward Processing Function ===
def process_csv_rewards(directory, step=10):
    """Read average reward per episode from a directory of CSVs."""
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return [], []

    csv_files = sorted(
        [file for file in os.listdir(directory) if file.endswith('.csv')],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])))
    )
    csv_files = csv_files[::step]

    if not csv_files:
        print(f"No CSV files found in: {directory}")
        return [], []

    rewards = []
    file_numbers = []

    for csv_file in csv_files:
        file_number = int(''.join(filter(str.isdigit, os.path.splitext(csv_file)[0])))
        file_numbers.append(file_number)

        file_path = os.path.join(directory, csv_file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        if "reward" not in df.columns:
            print(f"Missing 'reward' column in {file_path}")
            continue

        # Average reward (can be filtered to AVs only if needed)
        avg_reward = df["reward"].mean()
        rewards.append(avg_reward)

    return file_numbers, rewards


# === Plotting Function ===
def plot_metric_single_run(
    ax, directories, algorithm_names, process_function, colors,
    smoothing_window=10, show_legend=False, show_x_title=False,
    y_label="Average reward", legend_location='lower center'
):
    for i, directory in enumerate(directories):
        file_numbers, values = process_function(directory)

        if not values:
            continue

        smoothed_values = smooth_data(values, smoothing_window)
        if len(file_numbers) > len(smoothed_values):
            file_numbers = file_numbers[:len(smoothed_values)]

        ax.plot(
            file_numbers,
            smoothed_values,
            label=algorithm_names[i],
            color=colors[i % len(colors)],
            linewidth=1.5
        )

    ax.tick_params(axis='both', labelsize=8)
    ax.grid(visible=True, linestyle='--', linewidth=0.5)
    ax.set_xlabel("Episodes (days)", fontsize=12) if show_x_title else None
    ax.set_ylabel(y_label, fontsize=12)
    if show_legend:
        ax.legend(fontsize=7, loc=legend_location, ncol=2)


# === Plot Setup ===

directories = [
    "records_dqn/episodes",
    "records_ucb_two_route_net/episodes",
    "records_mbie_two_route_net/episodes",
    "records_rmax/episodes",
]

algorithm_names = [
    "DQN",
    "UCB",
    "MBIE",
    "Rmax"
]

colors = ["firebrick", "teal", "peru", "navy"]

# Create figure and axis
fig = plt.figure(figsize=(6, 3.5), dpi=300)
ax = fig.add_subplot(111)

# Plot the average rewards
plot_metric_single_run(
    ax=ax,
    directories=directories,
    algorithm_names=algorithm_names,
    process_function=process_csv_rewards,
    colors=colors,
    smoothing_window=2,
    show_legend=True,
    show_x_title=True,
    y_label="Average reward"
)

# Finalize and save
plt.tight_layout()
fig.canvas.draw()  # Ensure layout is set before saving
fig.savefig('avg_reward_algos_comp.png', dpi=300, bbox_inches='tight')
plt.show()
