import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS


def plot_d_svdd(data, model, sph_center, sph_radius, device):
    """
    Plots the DASVDD result showing support vectors, bounded support vectors, and the
    center in both latent space and data space based on given inputs. The method
    uses the model to transform the data to latent space and then visualizes the
    relationship between the transformed data points with respect to the hypersphere
    center and radius. If the data dimensionality exceeds two, multidimensional
    scaling (MDS) is applied for visualization in the data space.

    :param data: The input dataset to be processed and visualized
        in the latent space and data space. It should be in
        the form of a numpy array with samples as rows
        and features as columns.
    :param model: The trained model used to map the input dataset
        to the latent space. The model should be callable and
        return latent space embeddings.
    :param sph_center: The center of the hypersphere in the latent
        space. It is a PyTorch tensor and represents the center
        for calculating distances from transformed data points.
    :param sph_radius: The radius of the hypersphere in the latent
        space. It is a PyTorch tensor representing the boundary
        of normal data points.
    :param device: The device on which transformations (with the
        model) should be executed. Typically 'cpu' or 'cuda'.
    :return: None
    """

    model.eval()
    with torch.no_grad():
        phi_x = model.encoder(torch.tensor(data, dtype=torch.float32, device=device)).cpu().detach().numpy()

    center = sph_center.cpu().detach().numpy()
    radius = sph_radius.cpu().detach().numpy()
    dist_to_center = np.linalg.norm(phi_x - center, axis=1)
    is_bsv = dist_to_center >= radius
    is_sv = ((dist_to_center >= radius - 0.01) & (dist_to_center < radius))

    # Plot the latent space
    if sph_center.shape[0] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(phi_x[:, 0], phi_x[:, 1], label='ID')
        plt.scatter(phi_x[is_bsv, 0], phi_x[is_bsv, 1], c='red', label='BSV')
        plt.scatter(phi_x[is_sv, 0], phi_x[is_sv, 1], c='black', marker='s', label='SV')
        plt.scatter(center[0], center[1], c='orange', marker='*', s=100, label='Center')
        plt.legend()
        plt.show()

    # Plot the data space
    if data.shape[1] > 2:
        # Use MDS with the original data
        mds = MDS(n_components=2, random_state=42)
        x_mds = mds.fit_transform(data)

        plt.figure(figsize=(10, 8))
        plt.scatter(x_mds[:, 0], x_mds[:, 1], label='ID')
        plt.scatter(x_mds[is_bsv, 0], x_mds[is_bsv, 1], c='red', label='BSV')
        plt.scatter(x_mds[is_sv, 0], x_mds[is_sv, 1], c='black', marker='s', label='SV')
        plt.legend()
        plt.show()


def plot_training_loss(pd_results, fig_size=(10, 6)):
    """
    Visualizes the training and test loss over epochs.

    :param pd_results: A pandas dataframe containing 'epoch', 'train loss' and optionally 'test loss' columns
    :param fig_size: The figure dimensions (width, height). Defaults to (10, 6).
    """

    # Style configuration
    sns.set_style("whitegrid")

    # Figure setup
    plt.figure(figsize=fig_size)

    # Plot training loss
    plt.plot(
        pd_results["epoch"],
        pd_results["train loss"],
        label='Training Loss',
        marker='x'
    )

    # Add val loss line if available
    if "val loss" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["val loss"],
            label='Validation Loss',
            marker='o'
        )

    # Add a test loss line if available
    if "test loss" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["test loss"],
            label='Test Loss',
            marker='o'
        )

    # Configure labels and title
    plt.title('Loss Evolution per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_accuracy(pd_results, fig_size=(10, 6)):
    """
    Visualizes the model accuracy over epochs.

    :param pd_results: A pandas dataframe containing 'epoch', 'train Acc' and
        optionally 'test Acc' and 'val Acc' columns
    :param fig_size: The figure dimensions (width, height). Defaults to (10, 6).
    """

    # Style configuration
    sns.set_style("whitegrid")

    # Figure setup
    plt.figure(figsize=fig_size)

    # Plot training loss
    plt.plot(
        pd_results["epoch"],
        pd_results["train Acc"],
        label='Training Accuracy',
        marker='x'
    )

    # Add val loss line if available
    if "val Acc" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["val Acc"],
            label='Validation Accuracy',
            marker='o'
        )

    # Add a test loss line if available
    if "test Acc" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["test Acc"],
            label='Test Accuracy',
            marker='o'
        )

    # Configure labels and title
    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()