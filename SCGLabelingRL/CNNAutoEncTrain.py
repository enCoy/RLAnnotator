import numpy as np
import pickle
import os
from utils import get_standardizer, apply_standardizer
from torch.utils.data import Dataset, DataLoader
import torch
from Models import CnnAutoencoder
import torch.nn as nn
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def load_train_test_pigs(scg_dict, test_pigs, train_pigs, beat_length):
    # create an empty array with shape Nxbeat_length
    # N is the number of beats in the scg signal
    # beat_length is the length of the beat
    train_beats = np.zeros((0, beat_length))
    test_beats = np.zeros((0, beat_length))
    for pig in train_pigs:
        data = scg_dict[pig]
        # concatenate the data to the train_beats array
        train_beats = np.concatenate((train_beats, data[0]), axis=0)

    for pig in test_pigs:
        data = scg_dict[pig]
        # concatenate the data to the test_beats array
        test_beats = np.concatenate((test_beats, data[0]), axis=0)
    return train_beats, test_beats

if __name__ == "__main__":
    #todo: need to add validation portion later on

    machine = 'server'
    if machine == 'server':
        project_dir = r"/home/cmyldz/GaTech Dropbox/Cem Yaldiz/RLSCGLabeling"
    else:
        project_dir = r"C:\Users\Cem Okan\GaTech Dropbox\Cem Yaldiz\RLSCGLabeling"

    output_dir = os.path.join(project_dir, 'CNNAutoencoder')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # print used device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(os.path.join(project_dir, 'ProcessedData', 'padded_clipped_beat_dict.pkl'), "rb") as f:
        scg_dict = pickle.load(
            f)  # keys are subject ids, values are tuples (SCG_Z_padded signal, AO_labels, AC_labels)
    beat_length = 300
    pigs = [1, 2, 3, 4, 5, 6]
    for pig in pigs:
        pig_output_dir = os.path.join(output_dir, f'Pig{pig} As Test')
        if not os.path.exists(pig_output_dir):
            os.makedirs(pig_output_dir)

        test_pig = pig
        # train_pigs will be every pig except the test pig
        train_pigs = [p for p in pigs if p != test_pig]
        X_train, X_test = load_train_test_pigs(scg_dict, [test_pig], train_pigs, beat_length=beat_length)
        # downsample X_test by taking every 5th sample since we do not want to look at all of them
        X_test = X_test[::5]


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

        # create the dataloader
        train_dataset = TimeSeriesDataset(X_train)
        test_dataset = TimeSeriesDataset(X_test)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


        # initialize the model, loss function and optimizer
        model = CnnAutoencoder().cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        best_model_path = os.path.join(pig_output_dir, 'best_model.pth')
        num_epochs = 2

        # store losses for plotting
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for input in train_loader:
                input = input.cuda()
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, input)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    input = data.cuda()
                    output = model(input)
                    loss = criterion(output, input)
                    val_loss += loss.item()

            # track losses for plotting
            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss/len(test_loader))

            # check if the validation loss has improved
            avg_val_loss = val_loss / len(test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved with val loss: {best_val_loss:.4f}')

        print(f"Best model saved to {best_model_path}")

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(np.arange(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for Pig {pig}')
        plt.legend()
        plt.grid(True)

        # Save the plot
        loss_plot_path = os.path.join(pig_output_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")






