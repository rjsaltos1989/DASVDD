import matplotlib.pyplot as plt
import seaborn as sns


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