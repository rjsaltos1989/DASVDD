#----------------------------------------------------------------------------------
# Deep Autoencoding Support Vector Data Description (DASVDD)
#----------------------------------------------------------------------------------
# Author: Hadi Hojjati and Narges Armanfard (2024)
# Implementation: Ramiro Saltos Atiencia
# Date: 2025-06-01
# Version: 1.0
#----------------------------------------------------------------------------------

# Libraries
#----------------------------------------------------------------------------------

import os
import scipy.io as sio

from torch.utils.data import *
from svdd_nn_train_functions import *
from plot_functions import *
from svdd_eval_functions import *

# Setting up the device
#device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# %%Importing the data
#----------------------------------------------------------------------------------
dataset_file = 'shuttle.mat'
data_path = os.path.join('data', dataset_file)

# Load data
mat_data = sio.loadmat(data_path)
data = mat_data['Data']
labels = mat_data['y'].ravel() == 1

# Data dimensionality
num_obs, in_dim = data.shape


# %%Data Preparation
#----------------------------------------------------------------------------------

# Create a TensorDataset using the data
train_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                              torch.tensor(data, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                             torch.tensor(labels, dtype=torch.float32))

# Create a DataLoader for each dataset
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%Model Configuration
#----------------------------------------------------------------------------------
latent_dim = 3
layer_sizes = [in_dim, 7, 5, latent_dim]
gamma = set_gamma(layer_sizes, train_loader, device)
ae_model = AutoEncoder(layer_sizes)
ae_loss_fn = nn.MSELoss()

# %%Model Training
#----------------------------------------------------------------------------------

# Set the max epochs for training
train_epochs = 100

# Register the start time
start_time = time.time()

# Run the training phase
results_svdd_pd, sph_center = train_d_svdd_network(ae_model, ae_loss_fn, deep_svdd_loss, train_loader,
                                                   gamma=gamma, epochs=train_epochs, device=device)


# Register the end time
end_time = time.time()

print(f"Total training time was {end_time - start_time:.2f} seconds.")
print(f"Threads de OpenMP: {torch.get_num_threads()}")

# %%Plot the results
#----------------------------------------------------------------------------------
plot_training_loss(results_svdd_pd)

# %%Evaluate the performance
#----------------------------------------------------------------------------------

out_scores = get_outlier_scores(ae_model, test_loader, sph_center, gamma, device)
eval_metrics = eval_d_svdd(out_scores, labels)
print(eval_metrics)
