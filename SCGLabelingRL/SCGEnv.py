import pickle
import os
from VisualizerFunctions import ClickHandler, plot_time_series_matrix_with_selection, plot_with_selected_lines
import random
from SCGLabelingRL.Models import Actor
from scipy.signal import find_peaks

import numpy as np
from utils import get_boundary_constrain_reward, get_extremum_reward, get_consistency_reward, get_dtw_reward
from utils import z_score_normalize
from VisualizerFunctions import plot_time_series_matrix_signal_wise_normalization
import matplotlib.pyplot as plt
#todo: you asked chatgpt about choosing two points from the plot, integrate that
# todo: also instead of padding via finding the max, look at the distribution of lengths

class SCGEnv:
    def __init__(self, project_dir, scaler_path, scg_label_type, sampling_rate, downsampling_fac,
                 extremum_type='peak', num_past_detections=5, use_prominence=False, num_beats_in_episode=600):

        with open(os.path.join(project_dir, 'ProcessedData', 'padded_clipped_beat_dict.pkl'), "rb") as f:
            self.scg_dict = pickle.load(f)  # keys are subject ids, values are tuples (SCG_Z_padded signal, AO_labels, AC_labels)
        self.scg_label_type = scg_label_type   # this is required for evaluation, this is what we want to predict
        if self.scg_label_type == 'AO':
            self.label_index = 0
        else:
            self.label_index = 1  # AC
        self.beat_length = 296  # this number equals to the length of padded and clipped beat
        self.n_actions = self.beat_length  # we are going to estimate the sample index
        self.num_past_detections = num_past_detections
        self.num_beats_in_episode = num_beats_in_episode
        self.sampling_rate = sampling_rate
        self.downsampling_fac = downsampling_fac
        # state = scg_signal + (track_peak=0 1 if peak, 0 if valley) + prevKpredictions + boundaries
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # self.subject_based_boundaries = self.choose_intervals()
        self.subject_based_boundaries = {1: [4, 35], 2: [5, 40], 3: [5, 35], 4:[5, 30], 5: [5, 20], 6: [5, 30]}
        self.episode_dict, self.episode_label_dict, self.boundaries = self.create_episodes(episode_length=self.num_beats_in_episode, overlap_factor=0.5)
        self.num_episodes = len(list(self.episode_dict.keys()))
        self.episode_indices = [i for i in range(self.num_episodes)]
        self.last_episode_feedback_received = None
        self.last_x_choosen = None
        print("num keys: ", len(list(self.episode_dict.keys())))
        print("example shape: ", self.episode_dict[0].shape)
        print("example shape: ", self.episode_label_dict[0][0].shape)
        print("boundaries: ", self.boundaries)

        # these will be required for env.step
        self.current_episode = 0
        self.current_episode_step = 0
        self.current_state = None


        # reward function related params
        self.extremum_type = extremum_type
        self.use_prominence = use_prominence
        self.consistency_std_lim = 2

    def choose_intervals(self):
        subject_based_boundaries = {}
        # for sub_id in list(self.scg_dict.keys()):
        for sub_id in [6]:
            scg_subject = self.scg_dict[sub_id][0]
            # plot_time_series_matrix_signal_wise_normalization(scg_subject, upper_limit=250)

            # Initial plot for selecting points
            boundaries = plot_time_series_matrix_with_selection(scg_subject, sub_id, upper_limit=250, time_labels_interval=10)
            subject_based_boundaries[sub_id] = boundaries

            # call the following if you want to take a deeper look after you choose ao ac points
            # Plot with selected lines
            # plot_with_selected_lines(scg_subject, sub_id, x_values, upper_limit=250, time_labels_interval=10)
        return subject_based_boundaries

    def create_episodes(self, episode_length, overlap_factor):
        """
        Create episodes from time series data.

        Parameters:
        - data: Dictionary where keys are subject ids and values are lists of time series signals.
        - episode_length: Number of time series signals in a single episode.
        - overlap_factor: Overlapping factor between 0 and 1. Determines the stride for creating episodes.

        Returns:
        - episodes: Dictionary where keys are episode indices and values are lists of time series signals.
        """
        episodes = {}
        episode_labels = {}
        boundaries = {}

        episode_idx = 0
        stride = int(episode_length * (1 - overlap_factor))

        # for subject_id in list(self.scg_dict.keys()):
        for subject_id in [1, 2, 3, 4, 5, 6]:

            subject_selected_boundaries = self.subject_based_boundaries[subject_id]
            (scg_subject, ao_subject, ac_subject) = self.scg_dict[subject_id]
            scg_subject = scg_subject[:, :self.beat_length]  # we clip it to 296 to be able to use 3 layer CNN
            print("scg shape: ", scg_subject.shape)
            scg_subject = self.scaler.transform(scg_subject)
            num_series = len(scg_subject)
            start = 0

            while start + episode_length <= num_series:
                episode_scg = scg_subject[start:start + episode_length, :]
                episode_ao = ao_subject[start:start + episode_length]
                episode_ac = ac_subject[start:start + episode_length]

                episodes[episode_idx] = episode_scg
                boundaries[episode_idx] = subject_selected_boundaries
                episode_labels[episode_idx] = (episode_ao, episode_ac)
                episode_idx += 1
                start += stride

        return episodes, episode_labels, boundaries

    def reset(self):
        # new episode, get a random episode
        rand_episode = random.choice(self.episode_indices)
        # now take the very first beat from that episode
        beat = self.episode_dict[rand_episode][0]  # this should be length 1xBeatLength
        # take the corresponding boundaries
        boundary = np.array(self.boundaries[rand_episode])
        # now the list of previous detections will be list of -1s since no previous detection
        detections = np.array([-1 for j in range(self.num_past_detections)])
        observation = np.concatenate((beat, boundary, detections))
        self.current_episode = rand_episode
        self.current_episode_step = 0
        self.current_state = observation
        # concatenate all three
        return observation, rand_episode

    def step(self, action):
        # nearest_peak_x = None  # Local variable to store the nearest peak position
        # we are going to only change the state by selecting the next beat
        if self.current_episode_step + 1 < self.num_beats_in_episode:
            current_beat = self.current_state[:-self.num_past_detections-2]
            # take the detections
            detections = self.current_state[-self.num_past_detections:]  # last num past detections elements
            boundary = self.current_state[-self.num_past_detections - 2:-self.num_past_detections]

            boundary_coeff = 0.1
            extremum_coeff = 1
            consistency_coeff = 0.25

            boundary_reward = boundary_coeff * get_boundary_constrain_reward(action, boundary[0], boundary[1], slope=-1)
            extremum_reward = extremum_coeff * get_extremum_reward(current_beat, action, extremum_type=self.extremum_type, use_prominence=self.use_prominence)
            consistency_reward = consistency_coeff * get_consistency_reward(action, detections, std_lim=self.consistency_std_lim)

            # # Ask for human feedback by showing the signal and action as a vertical line at the corresponding x value
            # fig, ax = plt.subplots(figsize=(10, 5))
            # ax.plot(current_beat, label='Signal')
            # ax.axvline(action, color='r', linestyle='--', label='Action')
            # ax.set_xlabel('Time (beats)')
            # ax.set_ylabel('Signal Value')
            # ax.legend()
            # plt.title(f"Click on the nearest peak to the correct action - Episode {episode}")
            #
            # # Capture the click event to find the nearest peak
            # def onclick(event):
            #     nonlocal nearest_peak_x  # Allow modifying the nearest_peak_x variable
            #     # Find the x-coordinate of the click
            #     clicked_x = event.xdata
            #     if clicked_x is not None:
            #         # Find the nearest peak to the clicked point
            #         peaks, _ = find_peaks(current_beat)
            #         nearest_peak_idx = np.argmin(np.abs(peaks - clicked_x))
            #         nearest_peak_x = peaks[nearest_peak_idx]
            #         print(f"Nearest peak at x = {nearest_peak_x}")
            #
            #         # Save or process this information (e.g., return nearest peak x)
            #         # You can store this x position in a variable or use it for further calculations
            #         return nearest_peak_x
            # # Bind the click event
            # cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # plt.show()

            reward = boundary_reward + extremum_reward + consistency_reward
            rewards = {
                'boundary': boundary_reward,
                'extremum': extremum_reward,
                'consistency': consistency_reward,
                'total': reward
            }

            # update the state
            new_beat = self.episode_dict[self.current_episode][self.current_episode_step + 1]
            new_label = self.episode_label_dict[self.current_episode][self.label_index][self.current_episode_step + 1]
            # shift it to the left
            detections = np.roll(detections, -1)
            detections[-1] = action
            # boundaries will remain the same
            new_observation = np.concatenate((new_beat, boundary, detections))
            self.current_episode_step += 1
            done = False
            info = {'correct_label': new_label,
                    'boundary_orig': boundary_reward / boundary_coeff,
                    'extremum_orig': extremum_reward / extremum_coeff,
                    'consistency_orig': consistency_reward / consistency_coeff}   # for now only None

        else:
            new_observation = self.current_state
            rewards = {
                'boundary: ': None,
                'extremum: ': None,
                'consistency: ': None,
                'total': 0
            }
            done = True
            info = {'correct_label': None,
                    'boundary_orig': None,
                    'extremum_orig': None,
                    'consistency_orig': None,
                    }
        return new_observation, rewards, done, info









