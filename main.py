import numpy as np
import os
import torch
from submodular_functions.facility_location import make_kernel
import wandb
import sys

#wandb.init()
use_gpu = True if torch.cuda.is_available() else False
if use_gpu == False:
    raise ValueError("Can only proceed if gpu is available!")
data_path = os.path.join("./downloaded_data/", "data_set1.csv")

dataset = np.loadtxt(data_path, delimiter=",", dtype=float)
if use_gpu:
    dataset = torch.tensor(dataset).cuda()
print("shape of the dataset is", dataset.shape)
n_data = dataset.shape[0]
print(f"number of data points are {n_data}")
W  = make_kernel(dataset)

print("shape of the symmetric similarity kernel is ", W.shape)

# doing some initializations
V = np.arange(0, n_data)
print(V)
sys.exit()
greedy_max(dataset, )