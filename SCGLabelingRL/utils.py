import os
import torch
from torch.autograd import Variable
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def get_boundary_constrain_reward(action, lower_bound, upper_bound, slope=-1):
    """
    Calculate the reward based on whether the action is within the specified boundaries.

    Parameters:
    action (float): The action taken.
    lower_bound (float): The lower boundary.
    upper_bound (float): The upper boundary.
    slope (float): The slope of the penalty for actions outside the boundary.
                   Should be negative for decreasing penalty. Default is -1.

    Returns:
    float: The reward, which is 0 if the action is within the boundaries,
           and a negative value that decreases linearly with distance from the boundary otherwise.
    """

    if lower_bound <= action <= upper_bound:
        return 0
    else:
        if action < lower_bound:
            distance = lower_bound - action
        else:
            distance = action - upper_bound
        return slope * distance

def get_extremum_reward(signal, action, extremum_type='peak', use_prominence=False):
    """
    Calculate the reward based on whether the action corresponds to an extremum (peak or valley).

    Parameters:
    signal (np.ndarray): The signal array to analyze.
    action (int): The index of the action taken.
    extremum_type (str): Type of extremum to check for ('peak' or 'valley').
    use_prominence (bool): If true, considers prominence of extrema.

    Returns:
    float: The reward, which is 0 if the action corresponds to the desired extremum,
           and -1 * distance to the nearest or most prominent extremum otherwise.
    """

    # Invert the signal for valleys, since find_peaks finds peaks by default
    if extremum_type == 'valley':
        inverted_signal = -signal
    else:
        inverted_signal = signal

    # Find all peaks in the signal
    # plt.plot(signal)
    # plt.show()
    peaks, properties = find_peaks(inverted_signal, prominence=(0.005 if use_prominence else 0))  #todo: this parameter is manually adjusted for now
    if len(peaks) == 0:
        return -np.inf  # If no peaks or valleys are found, return a large negative reward

    if use_prominence:
        # Find the most prominent extremum
        most_prominent_idx = np.argmax(properties['prominences'])
        most_prominent_peak = peaks[most_prominent_idx]

        if action == most_prominent_peak:
            return 0
        else:
            return -abs(action - most_prominent_peak)
    else:
        # Find the closest extremum to the action
        closest_peak = peaks[np.argmin(abs(peaks - action))]

        if action == closest_peak:
            return 0
        else:
            return -abs(action - closest_peak)


def get_consistency_reward(action, previous_actions, std_lim=3):
    """
    Calculate the consistency reward based on the deviation from previously chosen actions.

    Parameters:
    action (float): The current action to evaluate.
    previous_actions (list): List of previously chosen actions, which may include None values.

    Returns:
    float: The reward, which is 0 if the action is within mean ± 3 * std deviation,
           and -1 otherwise. Returns 0 if all values are None.
    """
    # Convert to numpy array, handling None values
    filtered_actions = [x for x in previous_actions if x is not None]

    if len(filtered_actions) == 0:
        return 0  # All values were None

    previous_actions = np.array(filtered_actions, dtype=float)

    # Calculate the mean and standard deviation of previous actions
    mean = np.mean(previous_actions)
    std = np.std(previous_actions)

    # Define the acceptable range
    lower_bound = mean - std_lim * std
    upper_bound = mean + std_lim * std

    # Calculate the reward
    if lower_bound <= action <= upper_bound:
        return 0
    else:
        return -1

from dtaidistance import dtw
def get_dtw_reward(signal, signal_database):
    """
    Calculate the DTW-based reward for a given signal by finding the largest DTW score
    (smallest DTW distance) between the signal and each signal in the database.

    Parameters:
    - signal: array-like, the query signal for which the reward is calculated.
    - signal_database: list of array-like, each representing a signal in the database.

    Returns:
    - max_dtw_score: float, the largest DTW score (smallest DTW distance) found.
    """
    # Check if the signal is empty
    if signal.shape == (0,):
        return -100
    if signal.shape == (1,):
        return -100   #

    # z score normalization is needed
    signal = z_score_normalize(signal)
    # Calculate the initial DTW score (inverted distance) with the first signal in the database
    dtw_distance = dtw.distance(signal, signal_database[0])
    max_dtw_score = -dtw_distance  # Invert the distance for similarity score

    # Iterate through the rest of the signals in the database
    for db_signal in signal_database[1:]:
        dtw_distance = dtw.distance(signal, db_signal)
        dtw_score = -dtw_distance
        max_dtw_score = max(max_dtw_score, dtw_score)
    return max_dtw_score


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    print("aha parent dir: ", parent_dir)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

def z_score_normalize(signal, epsilon=1e-6):
    mean = np.mean(signal)
    std_dev = np.std(signal)
    normalized_signal = (signal - mean) / (std_dev + epsilon)
    return normalized_signal

def convert_from_sample_to_ms(value, sampling_rate, downsampling_factor):
    return value * downsampling_factor * 1000 / sampling_rate

from sklearn.cluster import AgglomerativeClustering
def cluster_signals(signal_database, n_clusters=10):
    # Compute pairwise DTW distances
    distance_matrix = np.array([[dtw.distance(s1, s2) for s2 in signal_database] for s1 in signal_database])

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    labels = clustering.fit_predict(distance_matrix)

    # Select one signal per cluster as the representative
    unique_signals = []
    for label in np.unique(labels):
        cluster_signals = [signal_database[i] for i in range(len(signal_database)) if labels[i] == label]
        unique_signals.append(
            cluster_signals[0])  # Take the first as representative; other selection strategies can also be used

    return unique_signals