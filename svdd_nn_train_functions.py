import torch
from torch import nn
from nn_models import AutoEncoder
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import time


def deep_svdd_loss(phi_x, sph_center):
    """
    Compute the One-class Deep SVDD loss.

    :param phi_x: a torch tensor with the projection of the input data onto the latent space.
    :param sph_center: a torch tensor with the center of the hypersphere.
    :return: loss: a torch tensor with the Deep SVDD loss.
    """

    # Compute the distances of phi(x,W) to the sphere center.
    dist = torch.sum((phi_x - sph_center) ** 2, dim=1)

    return torch.mean(dist)


def init_center(model, data_loader, device):
    """
    Initialize hypersphere center as the mean from an initial forward pass on the data.

    :param model: a Pytorch model.
    :param data_loader: a Pytorch DataLoader.
    :param device: a string specifying the device to use.
    :return sph_center: a torch tensor with the center of the hypersphere.
    """
    n_samples = 0
    sph_center = torch.zeros(model.latent_dim, device=device)

    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            phi_x = model.encoder(inputs)
            n_samples += phi_x.shape[0]
            sph_center += torch.sum(phi_x, dim=0)

    sph_center /= n_samples

    return sph_center


def set_gamma(layer_sizes, data_loader, device, num_T=10):
    """
    Calculate the gamma hyperparameter value by running an AutoEncoder model
    repeatedly using data from the provided data loader. Gamma is computed
    as the average of the reconstruction error divided by the distributional
    distance across multiple iterations.

    :param layer_sizes: List of integers representing the size of each layer in
        the AutoEncoder.
    :type layer_sizes: list[int]
    :param data_loader: DataLoader object providing the dataset to be used for
        training the AutoEncoder.
    :type data_loader: torch.utils.data.DataLoader
    :param device: The computation device (e.g., 'cpu' or 'cuda') on which the
        calculations are performed.
    :type device: torch.device
    :param num_T: Number of iterations to run the model and compute gamma.
        Default is 10.
    :type num_T: int
    :return: The computed gamma value as a float.
    :rtype: float
    """

    gamma = torch.tensor(0.0, device=device)
    for t in range(num_T):
        model = AutoEncoder(layer_sizes)
        model.to(device)
        rec_error = torch.tensor(0.0, device=device)
        dist = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                x_hat = model(inputs)
                phi_x = model.encoder(inputs)
                rec_error += nn.MSELoss()(inputs, x_hat)
                dist += nn.MSELoss(reduction='sum')(phi_x, torch.zeros_like(phi_x))

        rec_error /= len(data_loader)
        dist /= len(data_loader)
        gamma += rec_error / dist

    gamma /= num_T

    return gamma.item()


def run_d_svdd_epoch(model, optimizer, data_loader, ae_loss_fn, svdd_loss_fn, sph_center,
                     gamma, results, device, prefix=""):
    """
    Runs one epoch of training or testing for deep SVDD.

    Note: This functions has the side effect of updating the results dictionary.

    :param model: a Pytorch model.
    :param optimizer: a Pytorch optimizer.
    :param data_loader: a Pytorch DataLoader.
    :param ae_loss_fn: a Pytorch loss function. This is the autoencoder loss.
    :param svdd_loss_fn: a Pytorch loss function.
    :param sph_center: a torch tensor with the center of the hypersphere.
    :param gamma: a float indicating the weight of the SVDD loss in the total loss.
    :param results: a dictionary to store the results.
    :param device: a string specifying the device to use.
    :param prefix: a optional string to describe the results.
    :return: a float representing the total time taken for this epoch.
    """

    # Initialize some variables
    running_loss = []

    # Start the time counter
    start = time.time()

    # Loop over the batches in the data loader
    for inputs, labels in data_loader:

        # Moves the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass.
        x_hat = model(inputs)
        phi_x = model.encoder(inputs)

        # Compute loss.
        svdd_loss = svdd_loss_fn(phi_x, sph_center)
        ae_loss = ae_loss_fn(x_hat, labels)
        loss = ae_loss + gamma * svdd_loss

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save current loss
        running_loss.append(loss.item())

    # Stop the time counter
    end = time.time()

    # Compute the average loss for this epoch
    results[prefix + " loss"].append(np.mean(running_loss))

    # Return the time taken for this epoch
    return end - start


def train_d_svdd_network(model, ae_loss_fn, svdd_loss_fn, train_loader, gamma = 1,
                         init_lr=0.001, min_lr=0.0001, epochs=50, device='cpu',
                         lr_schedule=None, checkpoint_file=None):
    """
    Train a D-SVDD neural network using AdamW as a optimizer.

    Note: This functions has a side effect of saving the neural network training progress
    to a checkpoint file.

    :param model: a Pytorch model.
    :param ae_loss_fn: a Pytorch loss function for the autoencoder.
    :param svdd_loss_fn: a Pytorch loss function for the SVDD.
    :param train_loader: a DataLoader for the training set.
    :param gamma: a float indicating the weight of the SVDD loss in the total loss. Defaults to 1.
    :param init_lr: the initial learning rate. Defaults to 0.001.
    :param min_lr: the minimum learning rate. Defaults to 0.0001.
    :param epochs: the number of epochs to train for. Defaults to 50.
    :param device: a string specifying the device to use. Defaults to 'cpu'.
    :param lr_schedule: a string with learning rate schedule type. Defaults to None.
    :param checkpoint_file: a string specifying the checkpoint file to save. Defaults to None.
    :return: a pandas DataFrame containing the training and evaluation results.
    """

    # Initialize the information to be tracked
    to_track = ["epoch", "total time", "train loss"]

    # Keep track of the total training time
    total_train_time = 0

    # Initialize a dictionary to store the results
    results = {}

    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # Instantiate the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)

    # Instantiate the scheduler
    scheduler = None
    match lr_schedule:
        case "exp_decay":
            gamma = (min_lr / init_lr) ** (1 / epochs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        case "step_decay":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs // 4, gamma=0.3)

        case "cosine_decay":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 3, eta_min=min_lr)

        case "plateau_decay":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=0.2, patience=10)

        case _:
            pass

    # Move the model to the device
    model.to(device)

    # Initialize hypersphere center
    sph_center = init_center(model, train_loader, device)

    # Run the training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):

        # Put the model in training mode
        model = model.train()

        # Run an epoch of training
        total_train_time += run_d_svdd_epoch(model, optimizer, train_loader, ae_loss_fn, svdd_loss_fn, sph_center,
                                             gamma, results, device, prefix="train")

        # Update the sphere center every k epochs
        if epoch % 4 == 0:
            sph_center = init_center(model, train_loader, device)

        # Update the results
        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)

        # Update the learning rate after every epoch if provided
        if scheduler is not None:
            if lr_schedule == "plateau_decay":
                print("The plateau scheduler requires a validation loader to work.")
                break
            else:
                scheduler.step()

        # Save the results to a checkpoint file
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, checkpoint_file)

    # Return the results as a pandas DataFrame
    return pd.DataFrame.from_dict(results), sph_center