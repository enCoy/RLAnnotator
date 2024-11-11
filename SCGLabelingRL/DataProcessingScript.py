from SCGLabelingRL.Dataset import PigDataset



if __name__ == "__main__":
    project_dir = r"C:\Users\Cem Okan\GaTech Dropbox\Cem Yaldiz\RLSCGLabeling"
    dataset = PigDataset(data_dir=project_dir,
                 num_subjects=6,
                 sampling_rate=2000,
                 downsampling_fac=8, apply_filter=True)

    # dataset.visualize_random_signals(1, num_visualizations=7, window_sample_length=500)
    # dataset.visualize_random_signals(2, num_visualizations=7, window_sample_length=500)
    # dataset.visualize_random_signals(3, num_visualizations=3, window_sample_length=500)
    # dataset.visualize_random_signals(4, num_visualizations=3, window_sample_length=500)
    # dataset.visualize_random_signals(5, num_visualizations=7, window_sample_length=500)
    # dataset.visualize_random_signals(6, num_visualizations=3, window_sample_length=500)
