import numpy as np
import os
import torch
from submodular_functions.facility_location import make_kernel

use_gpu = True if torch.cuda.is_available() else False
if use_gpu == False:
    raise ValueError("Can only proceed if gpu is available!")
data_path = os.path.join("./downloaded_data/", "data_set1.csv")

dataset = np.loadtxt(data_path, delimiter=",", dtype=float)
if use_gpu:
    dataset = torch.tensor(dataset).cuda()
print("shape of the dataset is", dataset.shape)

W  = make_kernel(dataset)

print("shape of the symmetric similarity kernel is ", W.shape)
