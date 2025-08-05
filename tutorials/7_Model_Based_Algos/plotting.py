import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy import stats

def smooth_data(data, window_size):
    """Smooth data using a moving average with padding at the edges."""
    if len(data) < window_size:
        return data  # If the data is too short, return it unchanged
    half_window = window_size // 2
    # Pad the data at the edges to reduce edge effects
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    smoothed_data = np.convolve(
        padded_data, 
        np.ones(window_size) / window_size, 
        mode='valid'
    )
    # Ensure the result has the same length as the original data
    return smoothed_data[:len(data)]

# Function to process CSV files for a single algorithm
def process_csv_files(directory, step=10):
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return [], []

    csv_files = sorted(
        [file for file in os.listdir(directory) if file.endswith('.csv')],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
    
    # Select every `step`th file
    csv_files = csv_files[::step]
    
    if not csv_files:
        print(f"No CSV files found in: {directory}")
        return [], []

    action_proportions = []
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
        
        if "kind" not in df.columns or "action" not in df.columns:
            print(f"Missing expected columns in {file_path}")
            continue
        
        # Calculate the proportion of action 0 for kind = "AV"
        av_actions = df[df["kind"] == "AV"]["action"].value_counts(normalize=True).sort_index()
        action_proportions.append(av_actions.get(0, 0))  # Get proportion of action 0, default to 0

    
    return file_numbers, action_proportions


colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet"]

def plot_actions_single_run(
    ax, directories, algorithm_names, process_function, colors, 
    smoothing_window=10, show_legend=False, extend_x=0, 
    show_vertical_line=False, legend_location='lower center', show_x_title=False
):
    for i, directory in enumerate(directories):
        file_numbers, action_proportions = process_function(directory)

        if not action_proportions:
            continue

        # Smooth the proportion data
        smoothed_proportions = smooth_data(action_proportions, smoothing_window)

        # Match lengths
        if len(file_numbers) > len(smoothed_proportions):
            file_numbers = file_numbers[:len(smoothed_proportions)]

        ax.plot(
            file_numbers,
            smoothed_proportions,
            label=algorithm_names[i],
            color=colors[i % len(colors)],
            linewidth=1.5
        )

    # Horizontal line at y=1
    ax.axhline(
        y=1, color='black', linestyle='-', linewidth=4, label='Optimal Choice'
    )

    # Axes formatting
    ax.set_ylim(0, 1)
    #ax.set_xlim(0, 8300)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(visible=True, linestyle='--', linewidth=0.5)

    if show_x_title:
        ax.set_xlabel("Episodes (days)", fontsize=12, labelpad=10)

    if show_legend:
        ax.legend(fontsize=7, loc=legend_location, ncol=2)


import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

# Create a figure with 2 subplots
#fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300, sharey=True)

colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet"]

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy import stats

# Keep all function definitions as-is: smooth_data, process_csv_files, handle_outliers, process_csv_files_ones, etc.

# ... [All your existing function definitions go here] ...

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

# === Set up input data ===

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

colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet"]

# === Create a single plot ===

fig = plt.figure(figsize=(6, 3.5), dpi=300)
ax = fig.add_subplot(111)  # Single Axes

plot_actions_single_run(
    ax=ax,
    directories=directories,
    algorithm_names=algorithm_names,
    process_function=process_csv_files,
    colors=colors,
    smoothing_window=10,
    show_legend=True,
    extend_x=0,
    show_x_title=True
)

# Shared y-axis label
ax.set_ylabel("Fraction of AVs\nchoosing optimal action", fontsize=12)

plt.tight_layout()
plt.savefig('action_shifts_algos_comp.png', dpi=300)
