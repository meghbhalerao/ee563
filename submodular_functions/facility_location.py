from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn
import torch

def make_kernel(data, metric = 'euclidean', similarity = 'gaussian', sigma = 1):
    if metric == 'euclidean':
        W = torch.cdist(data, data, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

    else:
        raise ValueError(f"Entered {metric} metric, it is not supported yet!")

    # check which similarity to use
    if similarity == 'gaussian':
        W = torch.exp(-torch.pow(W,2)/sigma)
    
    elif similarity == 'linear':
        W = -W
    
    else:
        raise ValueError(f"Entered {similarity} similarity metric, it is not supported yet!")


    return W
def facility_location(V, A, W):
    pass
