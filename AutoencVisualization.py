import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import pickle
from SCGLabelingRL.HelperFunctions import load_train_test_pigs
from SCGLabelingRL.utils import get_standardizer, apply_standardizer
from SCGLabelingRL.Models import TimeSeriesDataset
from SCGLabelingRL.Models import CnnAutoencoder
from torch.utils.data import Dataset, DataLoader


def plot_reconstruction(X_test, model, device, num_samples=9):
    """
    Function to randomly sample 'num_samples' indices from the test set and plot reconstruction vs ground truth.

    Parameters:
        X_test (np.array): The test set data, shape (N, T, 1).
        model (torch.nn.Module): The trained autoencoder model.
        device (torch.device): The device to run the model on (CPU or GPU).
        num_samples (int): Number of random samples to show (default 9).
    """
    # Randomly sample indices
    random_indices = np.random.choice(X_test.shape[0], size=num_samples, replace=False)

    # Create a figure with subplots (3 rows and 3 columns)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # Loop through the samples and plot the ground truth and reconstruction
    for i, idx in enumerate(random_indices):
        # Get the ground truth signal
        original_signal = X_test[idx, :, 0]  # shape (T,)

        # Prepare the signal for the model (add batch dimension)
        input_signal = torch.tensor(original_signal).float().unsqueeze(0).unsqueeze(-1).to(device)
        # Pass the input through the model to get the reconstruction
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            reconstructed_signal = model(input_signal).cpu().numpy().squeeze()  # Remove extra dimensions

        # Plot original vs reconstructed signal
        ax = axs[i // 3, i % 3]
        ax.plot(original_signal, label='Ground Truth', color='blue')
        ax.plot(reconstructed_signal, label='Reconstruction', color='red', linestyle='dashed')
        ax.set_title(f"Sample {idx}")
        ax.legend()
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Signal Amplitude")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    machine = 'local'
    if machine == 'server':
        project_dir = r"/home/cmyldz/GaTech Dropbox/Cem Yaldiz/RLSCGLabeling"
    else:
        project_dir = r"C:\Users\Cem Okan\GaTech Dropbox\Cem Yaldiz\RLSCGLabeling"

    # print used device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(os.path.join(project_dir, 'ProcessedData', 'padded_clipped_beat_dict.pkl'), "rb") as f:
        scg_dict = pickle.load(
            f)  # keys are subject ids, values are tuples (SCG_Z_padded signal, AO_labels, AC_labels)
    beat_length = 300
    pigs = [1, 2, 3, 4, 5, 6]
    for pig in pigs:


        test_pig = pig
        # train_pigs will be every pig except the test pig
        train_pigs = [p for p in pigs if p != test_pig]

        test_pig_model_state_path = os.path.join(project_dir, 'CNNAutoencoder', f'Pig{test_pig} As Test', 'best_model.pth')

        X_train, X_test = load_train_test_pigs(scg_dict, [test_pig], train_pigs, beat_length=beat_length)
        # downsample X_test by taking every 5th sample since we do not want to look at all of them
        X_test = X_test[::7]


        # to be able to use 3 halving
        X_train = X_train[:, :296]
        X_test = X_test[:, :296]

        # add extra 1 dimension to convert it from NxT to NxTxF where F=1 in this problem
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        # get the standardizer from X_train and apply it to X_train and X_test
        standardizer = get_standardizer(X_train)
        X_train = apply_standardizer(X_train, standardizer)
        X_test = apply_standardizer(X_test, standardizer)

        # initialize the model, loss function and optimizer
        model = CnnAutoencoder().cuda()
        # load model state
        model.load_state_dict(torch.load(test_pig_model_state_path))


        plot_reconstruction(X_test, model, device)

