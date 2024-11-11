import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json
import scipy.io as sio
import mat73
from HelperFunctions import downsample_signal, create_kaiser_BPF_for_signals, peak_idx_correction
from HelperFunctions import save_dict_and_description
from VisualizerFunctions import generate_beat_length_histogram
from collections import defaultdict
from scipy.signal import filtfilt

class PigDataset():
    def __init__(self, data_dir,
                 num_subjects=6,
                 sampling_rate=2000,
                 downsampling_fac=8, apply_filter=False):

        # reading and writing folders
        self.dataset_dir = os.path.join(data_dir, 'RawData')   # data directory (dropbox etc)
        self.result_dir = os.path.join(data_dir, 'ResultsAndFigures', 'DataProcessing')
        # dataset specific parameters
        self.num_subjects = num_subjects  # number of subjects/participants in the dataset
        self.sampling_rate = sampling_rate  # in Hz
        self.downsampling_fac = downsampling_fac  # factor of downsampling for ECG and SCG waveforms, if None no downsampling
        self.apply_filter = apply_filter  # True if you want to apply filter


        self.processed_data_dir = os.path.join(data_dir, 'ProcessedData')
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

        self.load_and_process_raw_data()



    def load_and_process_raw_data(self):
        self.dict_subjects = self.load_data()
        print("Dictionary of Subjects is created!")
        # self.segment_raw_data_into_windows()
        print("Segmentation of the signal beat-by-beat starts!")
        beat_by_beat_dict = self.segment_beat_by_beat()
        print("Segmentation of the signal beat-by-beat ends!")
        # visualize histogram of beats
        generate_beat_length_histogram(beat_by_beat_dict, output_dir=self.result_dir, show_plot=False)
        padded_clipped_dict = self.pad_clip_beats(beat_by_beat_dict)


    def load_data(self):
        dict_subjects = {}
        # keys are subject ids, values are dictionaries with 'data'-scg_z, 'ao_label', 'ac_label'
        for i in range(1, self.num_subjects + 1):
            dict_subjects[i] = {
                'ecg': np.zeros(0),
                'scg_z': np.zeros(0),
                'label_ao': np.zeros(0),
                'label_ac': np.zeros(0),
                'label_r': np.zeros(0)
            }
        for i in range(1, self.num_subjects + 1):
            data_path = os.path.join(self.dataset_dir, f'pig_{i}_ecg_scg_phase_3' + '.mat')
            mat_data = mat73.loadmat(data_path)
            # read ecg
            ecg_unsegmented = mat_data['ecg']
            r_peak_timings = np.reshape(mat_data['r_peaks_new'], (-1, 1))
            # segment ecg using r-peak timings
            ecg = ecg_unsegmented[int(r_peak_timings[0]):int(r_peak_timings[-1])]
            # read scg
            scg_z = mat_data['scgz']  # relative to rr timings
            scg_z = scg_z[int(r_peak_timings[0]):int(r_peak_timings[-1])]
            # scg ao ac timings
            scg_ac_rel_timings = np.reshape(mat_data['rac'], (-1, 1))  # relative to rr timings
            scg_ao_rel_timings = np.reshape(mat_data['rao'], (-1, 1))  # relative to rr timings
            # subtract the initial time to make them start at 0
            r_peak_time_references = r_peak_timings - r_peak_timings[0]
            # consistency correction for pig1_ao, pig2_ao, pig5_ac
            # consistency in this case means that for the deep learning algorithm, we always select peaks or valleys
            # and do not change them across pigs
            if i in [1, 2, 5]:
                if i == 1 or i == 2:
                    consistency_data_path = os.path.join(self.dataset_dir, f'updated_feats' + '.mat')
                    consistency_mat_data = sio.loadmat(consistency_data_path)
                    scg_ao_rel_timings = np.reshape(consistency_mat_data[f'pig{i}_ao'], (-1, 1))
                    scg_ao_rel_timings = np.reshape(np.append(scg_ao_rel_timings, scg_ao_rel_timings[-1]), (-1, 1))
                    scg_ao_abs_timings = scg_ao_rel_timings + r_peak_time_references
                    scg_ac_abs_timings = 2 * scg_ac_rel_timings + r_peak_time_references

                if i == 5:
                    consistency_data_path = os.path.join(self.dataset_dir, f'updated_feats' + '.mat')
                    consistency_mat_data = sio.loadmat(consistency_data_path)
                    scg_ac_rel_timings = np.reshape(consistency_mat_data[f'pig{i}_ac'], (-1, 1))
                    scg_ac_rel_timings = np.reshape(np.append(scg_ac_rel_timings, scg_ac_rel_timings[-1]), (-1, 1))
                    scg_ao_abs_timings = 2 * scg_ao_rel_timings + r_peak_time_references
                    scg_ac_abs_timings = scg_ac_rel_timings + r_peak_time_references
            else:
                scg_ao_abs_timings = 2 * scg_ao_rel_timings + r_peak_time_references
                scg_ac_abs_timings = 2 * scg_ac_rel_timings + r_peak_time_references

            if self.downsampling_fac is not None:  # if you want to downsample
                ecg = downsample_signal(ecg, self.downsampling_fac)
                scg_z = downsample_signal(np.squeeze(scg_z), self.downsampling_fac)
                # we have to update timings accordingly since those contain values before considering downsampling
                r_peak_time_references = np.floor(np.array(r_peak_time_references) / self.downsampling_fac).astype(int)
                scg_ao_abs_timings = np.floor(np.array(scg_ao_abs_timings) / self.downsampling_fac).astype(int)
                scg_ac_abs_timings = np.floor(np.array(scg_ac_abs_timings) / self.downsampling_fac).astype(int)

            if self.apply_filter:
                # filter data
                filter = create_kaiser_BPF_for_signals(Fs=self.sampling_rate // self.downsampling_fac,
                                                            lower_cutoff=5, higher_cutoff=40, order=20,
                                                            filter_type='bandpass', scale=True, kaiser_beta=0.5)
                ecg = filtfilt(filter, 1, ecg)
                scg_z = filtfilt(filter, 1, scg_z)

            # correction of AO/AC peak idxes (since David's algorithm's return is not spot on but very close to real peaks)
            # num neighbors are tuned manually
            scg_ao_abs_timings = peak_idx_correction(scg_z, scg_ao_abs_timings, num_neighbors=10)
            scg_ac_abs_timings = peak_idx_correction(scg_z, scg_ac_abs_timings, num_neighbors=5)
            r_peak_time_references = r_peak_time_references[:-1]

            # concat and reshape
            ecg = np.reshape(ecg, (-1, 1))
            scg_z = np.reshape(scg_z, (-1, 1))

            dict_subjects[i]['ecg'] = ecg
            dict_subjects[i]['scg_z'] = scg_z
            dict_subjects[i]['label_ao'] = scg_ao_abs_timings
            dict_subjects[i]['label_ac'] = scg_ac_abs_timings
            dict_subjects[i]['label_r'] = r_peak_time_references
        return dict_subjects


    def segment_beat_by_beat(self):
        segmented_data_dict = {}  # keys are subject ids,
        # values are 'data' and 'labels'
        for i in range(1, self.num_subjects + 1):
            # in the following the most inner dictionary's key will be beat index
            segmented_data_dict[i] = {
                'scg_z': {},
                'label_ao': {},
                'label_ac': {}
            }

        total = 0
        for i in range(1, self.num_subjects + 1):
            r_peak_timings = self.dict_subjects[i]['label_r']
            scg_data = self.dict_subjects[i]['scg_z']
            ao_timings = self.dict_subjects[i]['label_ao']
            ac_timings = self.dict_subjects[i]['label_ac']

            counter = 0
            for j in range(len(r_peak_timings) - 1):
                start_idx = np.squeeze(r_peak_timings)[j]
                end_idx = np.squeeze(r_peak_timings)[j+1]
                scg_beat = scg_data[start_idx:end_idx, :]

                # note that ao and ac labels are relative to starting R-peak-timing
                ao_label = np.squeeze(ao_timings[j]) - start_idx
                ac_label = np.squeeze(ac_timings[j]) - start_idx

                segmented_data_dict[i]['scg_z'][j] = scg_beat
                segmented_data_dict[i]['label_ao'][j] = ao_label
                segmented_data_dict[i]['label_ac'][j] = ac_label
                counter += 1

            print(f"Pig {i} Num Beats = {counter}")
        print("In total Num samples: ", total)
        save_dict_and_description(segmented_data_dict, 'beat_by_beat_data_dict_varying_beat_length', self.processed_data_dir,
                                  description='There are 6 pigs in this dataset, and for each pig, we have SCG_z signal\n'
                                              'as well as Aortic Opening (AO)/Aortic Closing (AC) timings. \n These are all stored'
                                              'inside this dictionary. \n  Note that each beat might have different length. Use'
                                              'scg_z data for your RL algorithm and utilize labels as human feedback if you wish')
        return segmented_data_dict

    def pad_clip_beats(self, beat_by_beat_dict, threshold=300):
        # threshold is found by looking at the generated histograms

        # Step 2: Clip each time series to the minimum length
        new_data_dict  = {}
        for subject_id, beats in beat_by_beat_dict.items():
            new_time_series = []
            ao_labs = []
            ac_labs = []

            num_keys = len(list(beats['scg_z'].keys()))
            for beat_idx in range(num_keys):
                time_series = np.squeeze(beats['scg_z'][beat_idx])
                time_series_length = len(time_series)

                if time_series_length < threshold:
                    # pad the time series
                    new_time_series.append(np.pad(time_series, (0, threshold - len(time_series)), 'edge'))
                else:
                    # clip the time series
                    new_time_series.append(time_series[:threshold])

                ao_labs.append(beats['label_ao'][beat_idx])
                ac_labs.append(beats['label_ac'][beat_idx])

            # Convert lists to numpy arrays and store them in the new dictionary
            new_data_dict[subject_id] = (
                np.array(new_time_series),
                np.array(ao_labs),
                np.array(ac_labs)
            )
        save_dict_and_description(new_data_dict, 'padded_clipped_beat_dict',
                                  self.processed_data_dir,
                                  description='The keys of this dict are subject numbers.\nEach value'
                                              'is a tuple of (scg_z_signal, ao_labels, ac_labels)\nFor padding'
                                              f'edge value is repeated. If the signal has more than {threshold} '
                                              f'time steps, it is clipped.')
        return new_data_dict


    def visualize_random_signals(self, subject_id, num_visualizations=10, window_sample_length=500):
        """
        creates a subplot showing fiducial points of scg and ecg signals
        used to debug the code and make sure that everything is working fine
        :param subject_id:
        :param num_visualizations:
        :param window_sample_length:
        :return:
        """
        ecg_waveform = np.squeeze(self.dict_subjects[subject_id]['ecg'])
        r_peaks = self.dict_subjects[subject_id]['label_r']
        scg_z = np.squeeze(self.dict_subjects[subject_id]['scg_z'])
        ao = self.dict_subjects[subject_id]['label_ao']
        ac = self.dict_subjects[subject_id]['label_ac']

        # draw a random initial point
        signal_length = scg_z.shape[0]
        starting_point = np.random.randint(low=0, high=signal_length - window_sample_length - 5, size=num_visualizations)
        for idx in starting_point:
            scg_seg = scg_z[idx:idx + window_sample_length]
            ecg_seg = ecg_waveform[idx:idx + window_sample_length]

            fig, axs = plt.subplots(2, sharex='all')
            fig.suptitle(f'SCG-Z and ECG Signals - Pig {subject_id} - Random Sample Idx: {idx}')
            axs[0].plot(np.arange(len(scg_seg)), scg_seg)
            axs[1].plot(np.arange(len(ecg_seg)), ecg_seg)

            # now, retrieve r_peak, ao/ac timings between start and end indices
            r_peak_idx = np.bitwise_and(r_peaks >= idx, r_peaks <= idx+window_sample_length)
            r_peaks_window = r_peaks[r_peak_idx]
            scg_ao_idx = np.bitwise_and(ao >= idx, ao <= idx+window_sample_length)
            scg_ac_idx = np.bitwise_and(ac >= idx, ac <= idx+window_sample_length)
            scg_ao_window = ao[scg_ao_idx]
            scg_ac_window = ac[scg_ac_idx]
            if len(r_peaks_window) > 0:
                for j in range(len(r_peaks_window)):
                    # put dots to corresponding positions
                    x_axis_pos = int(r_peaks_window[j] - idx)
                    axs[1].plot(x_axis_pos, ecg_seg[x_axis_pos],
                                marker="*", markersize=12, markeredgecolor="red", markerfacecolor="red")
            if  len(scg_ao_window) > 0:
                for j in range(len(scg_ao_window)):
                    x_axis_ao_pos = int(scg_ao_window[j] - idx)
                    axs[0].plot(x_axis_ao_pos, scg_seg[x_axis_ao_pos],
                                marker="o", markersize=12, markeredgecolor="tab:green", markerfacecolor="tab:green")
            if len(scg_ac_window) > 0:
                for j in range(len(scg_ac_window)):
                    x_axis_ac_pos = int(scg_ac_window[j] - idx)
                    axs[0].plot(x_axis_ac_pos, scg_seg[x_axis_ac_pos],
                                marker="*", markersize=12, markeredgecolor="black", markerfacecolor="tab:green")
            plt.show()


