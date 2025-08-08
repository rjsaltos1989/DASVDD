# Simplified PyTorch Implementation of Deep Autoencoding SVDD

This [PyTorch](https://pytorch.org/) implementation of the *Deep Autoencoding Support Vector Data Description* (DASVDD) method is a simplified version based on the paper by Hadi Hojjati and Narges Armanfard (doi: 10.1109/TKDE.2023.3328882). The original DASVDD implementation is available at [Armanfard-Lab/DASVDD](https://github.com/Armanfard-Lab/DASVDD).

## Overview

Deep Autoencoding Support Vector Data Description (DASVDD) is an unsupervised anomaly detection algorithm that combines the strengths of autoencoders and Support Vector Data Description (SVDD). The algorithm uses a modified autoencoder architecture with a hypersphere constraint in the latent space to better detect anomalies. Note that DASVDD was only formulated for the one-class SVDD.

The project includes:
- Implementation of the DASVDD method based on the Hojjati and Armanfard (2023) paper
- Modified autoencoder architecture with hypersphere constraint
- Evaluation metrics for anomaly detection performance

## Algorithm Description

The DASVDD algorithm combines autoencoder reconstruction with a hypersphere constraint in the latent space:

1. **Training Phase**: The autoencoder is trained with a combined loss function that includes:
   - Reconstruction loss: Measures how well the autoencoder can reconstruct the input data
   - Hypersphere constraint: Encourages the latent representations to be close to a predefined center in the latent space

2. **Anomaly Detection Phase**: Anomalies are detected based on a combined score that considers both:
   - Reconstruction error: How well the input can be reconstructed
   - Distance from center: How far the latent representation is from the center of the hypersphere

The objective function for DASVDD combines these components to create a more robust anomaly detection method that leverages both the reconstruction capabilities of autoencoders and the compact representation of SVDD.

## Requirements
- Python 3.12
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.2.0
- tqdm >= 4.65.0
- SciPy >= 1.10.0

All dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rjsaltos1989/DASVDD.git
   cd DASVDD
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n dasvdd python=3.12
   conda activate dasvdd
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `main.py`: Main script to run the DASVDD algorithm
- `nn_models.py`: Neural network model definitions (AutoEncoder and DASVDD models)
- `svdd_nn_train_functions.py`: Functions for training the DASVDD model
- `svdd_eval_functions.py`: Functions for evaluating the DASVDD model
- `plot_functions.py`: Functions for visualizing results

## Usage

### Basic Usage

1. Modify the dataset path in `main.py` to point to your data:
   ```python
   dataset_path = '/path/to/your/data/'
   dataset_file = 'YourDataset.mat'
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

### Customization

You can customize the following parameters in `main.py`:

- `latent_dim`: Dimension of the latent space (default: 2)
- `train_epochs`: Number of epochs for DASVDD training (default: 100)
- `batch_size`: Batch size for training (default: 32)
- `gamma`: Weight parameter balancing reconstruction loss and hypersphere constraint

### Input Data Format

The code expects data in MATLAB .mat format with:
- 'Data': Matrix where rows are samples and columns are features
- 'y': Vector of labels where anomalies are labeled as 2

## Example Results

When running the code, you'll get:
1. Training loss plots showing the convergence of the model, including both reconstruction loss and hypersphere constraint components
2. Visualization of the data in the latent space, showing:
   - Normal data points
   - Anomalous data points
   - The center of the hypersphere
3. Performance metrics including AUC-ROC, AUC-PR, F1-Score, and Recall

## References

- DASVDD Paper: Hojjati, H., & Armanfard, N. (2023). Deep Autoencoding Support Vector Data Description for Unsupervised Anomaly Detection. IEEE Transactions on Knowledge and Data Engineering. doi: 10.1109/TKDE.2023.3328882
- Original DASVDD Implementation: [Armanfard-Lab/DASVDD](https://github.com/Armanfard-Lab/DASVDD)
- Deep SVDD Paper: Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, E., & Kloft, M. (2018). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
