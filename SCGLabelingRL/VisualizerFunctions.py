import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class ClickHandler:
    def __init__(self, ax, num_clicks=2):
        self.num_clicks = num_clicks
        self.clicks = []
        self.ax = ax
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)
        self.lines = []

    def __call__(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            self.clicks.append((event.xdata, event.ydata))
            line = self.ax.axvline(x=event.xdata, color='r', linestyle='--')
            self.lines.append(line)
        elif event.button == 3 and self.clicks:  # Right click and there are lines to remove
            self.clicks.pop()
            line = self.lines.pop()
            line.remove()

        self.ax.figure.canvas.draw()
        if len(self.clicks) == self.num_clicks:
            plt.close(self.ax.figure)

    def get_clicks(self):
        return self.clicks


def plot_time_series_matrix_with_selection(data, sub_id, upper_limit, time_labels_interval=10):
    """
    Plots N time series data in a 2D matrix with grayscale values.
    Each time series is normalized separately and allows for selecting two points to draw vertical lines.

    Parameters:
    - data: A list of time series signals (each a list of numerical values).
    - time_labels_interval: Interval at which to display time labels on the x-axis.
    """
    # Normalize each time series signal separately
    normalized_data = []
    for series in data:
        series = np.array(series)[:upper_limit]
        norm_series = (series - series.min()) / (series.max() - series.min())
        normalized_data.append(norm_series)

    # Convert the normalized list of time series signals into a 2D numpy array
    norm_matrix = np.array(normalized_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    click_handler = ClickHandler(ax, num_clicks=2)
    sns.heatmap(norm_matrix, cmap='gray', cbar=True, yticklabels=False, xticklabels=True, ax=ax)

    # Set x-axis labels at specified intervals
    ax.set_xticks(np.arange(0, norm_matrix.shape[1], time_labels_interval))
    ax.set_xticklabels(np.arange(0, norm_matrix.shape[1], time_labels_interval))

    ax.set_title(f"Time Series Data in 2D Matrix - Sub: {sub_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")

    # Display the plot and wait for clicks
    plt.show(block=False)
    while len(click_handler.get_clicks()) < click_handler.num_clicks:
        plt.waitforbuttonpress()

    # Retrieve x-values of the selected points
    points = click_handler.get_clicks()
    x_values = [int(point[0]) for point in points]
    print("Selected x-values:", x_values)

    return x_values


def plot_with_selected_lines(data, sub_id, x_values, upper_limit, time_labels_interval=10):
    """
    Plots N time series data in a 2D matrix with grayscale values and vertical lines at specified x-values.

    Parameters:
    - data: A list of time series signals (each a list of numerical values).
    - x_values: List of x-values where vertical lines should be drawn.
    - time_labels_interval: Interval at which to display time labels on the x-axis.
    """
    # Normalize each time series signal separately
    normalized_data = []
    for series in data:
        series = np.array(series)[:upper_limit]
        norm_series = (series - series.min()) / (series.max() - series.min())
        normalized_data.append(norm_series)

    # Convert the normalized list of time series signals into a 2D numpy array
    norm_matrix = np.array(normalized_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(norm_matrix, cmap='gray', cbar=True, yticklabels=False, xticklabels=True, ax=ax)

    # Set x-axis labels at specified intervals
    ax.set_xticks(np.arange(0, norm_matrix.shape[1], time_labels_interval))
    ax.set_xticklabels(np.arange(0, norm_matrix.shape[1], time_labels_interval))

    ax.set_title(f"Time Series Data in 2D Matrix with Selected Lines - Sub: {sub_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")

    # Draw vertical lines at the selected x-values
    for x in x_values:
        ax.axvline(x=x, color='r', linestyle='--')

    # Display the plot
    plt.show()



### old
def plot_time_series_matrix_glob_normalization(data):
    """
    Plots N time series data in a 2D matrix with grayscale values.

    Parameters:
    - data: A list of time series signals (each a list of numerical values).
    """
    # Convert the list of time series signals into a 2D numpy array
    matrix = np.array(data)

    # Normalize the data to [0, 1] range for grayscale representation
    norm_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_matrix, cmap='gray', cbar=True, yticklabels=False, xticklabels=False)
    plt.title("Time Series Data in 2D Matrix")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()


def plot_time_series_matrix_signal_wise_normalization(data, upper_limit, time_labels_interval=10):
    """
    Plots N time series data in a 2D matrix with grayscale values.
    Each time series is normalized separately.

    Parameters:
    - data: A list of time series signals (each a list of numerical values).
    - time_labels_interval: Interval at which to display time labels on the x-axis.
    """
    # Normalize each time series signal separately
    normalized_data = []
    for series in data:
        series = np.array(series[:upper_limit])
        norm_series = (series - series.min()) / (series.max() - series.min())
        normalized_data.append(norm_series)

    # Convert the normalized list of time series signals into a 2D numpy array
    norm_matrix = np.array(normalized_data)

    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(norm_matrix, cmap='gray', cbar=True, yticklabels=False, xticklabels=True)

    # Set x-axis labels at specified intervals
    ax.set_xticks(np.arange(0, norm_matrix.shape[1], time_labels_interval))
    ax.set_xticklabels(np.arange(0, norm_matrix.shape[1], time_labels_interval))

    plt.title("Time Series Data in 2D Matrix")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()

    print("Please click two points on the plot to select vertical lines.")
    points = plt.ginput(2)

    # Retrieve x-values of the selected points
    x_values = [int(point[0]) for point in points]
    print("Selected x-values:", x_values)
    return x_values




#####
def generate_beat_length_histogram(beat_by_beat_dict, output_dir=None, show_plot=True):
    """

    :param beat_by_beat_dict: beat_by_beat_dict[SubjectID] = {
                'scg_z': {},
                'label_ao': {key=beat_idx},
                'label_ac': {}
            }
    :return:
    """

    # Step 1: Find the minimum length of time series across all subjects and beats
    lengths = {}
    # initialize lists
    subjects = list(beat_by_beat_dict.keys())

    for subject_id in subjects:
        lengths[subject_id] = []

    for subject_id, beats in beat_by_beat_dict.items():
        for beat_index, beat_data in beats['scg_z'].items():
            time_series_length = len(beat_data)
            lengths[subject_id].append(time_series_length)

    # now visualization of histogram of beat lengths
    # Generate 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Histograms of Lengths by Subject')
    for i, ax in enumerate(axs.flat):
        if i < len(subjects):
            subject = subjects[i]
            length_sub = lengths[subject]
            ax.hist(length_sub, bins=20, alpha=0.75)
            ax.set_title(f'Histogram for {subject}')
            ax.set_xlabel('Length')
            ax.set_ylabel('Frequency')
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_dir is not None:
        lengths_subplot_filename = os.path.join(output_dir, 'lengths_each_pig_histogram.png')
        plt.savefig(lengths_subplot_filename, dpi=600)

    # Generate separate histogram for all lengths combined
    all_lengths = np.concatenate(list(lengths.values()))

    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=20, alpha=0.75)
    plt.title('Histogram of All Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    if output_dir is not None:
        all_lengths_filename = os.path.join(output_dir, 'all_beat_lengths_histogram.png')
        plt.savefig(all_lengths_filename, dpi=600)

    if show_plot:
        plt.show()