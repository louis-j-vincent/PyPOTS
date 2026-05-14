import torch
import numpy as np
import matplotlib.pyplot as plt

from GP_regression_helpers import *
from pygrinder import mcar


def artifical_GP_generation(n_observations_per_group = 1000,
                            n_dimensions = 5,
                            n_time_points = 30,
                            n_new_dims = 5,
                            p_dataset = 0.5):

    # Parameters
    n_groups = 1
    time_points = np.linspace(0, 10, n_time_points)
    length_scale = 1.5

    # Generate GP samples
    gp_samples = generate_gp_samples_with_library(
        n_groups=n_groups,
        n_observations_per_group=n_observations_per_group,
        n_dimensions=n_dimensions,
        time_points=time_points,
        length_scale=length_scale
    )

    Proj = np.random.uniform(size = (n_dimensions, n_new_dims))#.reshape(1,1,n_dimensions, n_new_dims)
    X_ori = gp_samples[0] @ Proj

    # Normalize

    X_normalized = (X_ori - X_ori.mean(axis=(0,1)).reshape(1,1,-1)) / X_ori.std(axis=(0,1)).reshape(1,1,-1)[:1000]
    X = mcar(X_normalized.copy(), p_dataset)  # randomly hold out 10% observed values as ground truth
    X_nan = X.copy()
    X[X!=X] = 0
    dataset = {'X' : X}

    len_dataset = len(dataset['X'])
    cut = int(len_dataset*0.7)
    dataset_train, dataset_val = {'X':dataset['X'][:cut]}, {'X':dataset['X'][cut:], 'X_ori':X_normalized[cut:]}

    dataset_nan = {'X' : X_nan}
    dataset_train_nan, dataset_val_nan = {'X':dataset_nan['X'][:cut]}, {'X':dataset_nan['X'][cut:], 'X_ori':X_normalized[cut:]}

    # add masks
    for dict in [dataset_train, dataset_val]:
        missing_mask = (dict['X']!=0)
        dict['missing_mask'] = missing_mask

    # add masks
    for dict in [dataset_train_nan, dataset_val_nan]:
        missing_mask = (dict['X']==dict['X'])
        dict['missing_mask'] = missing_mask

    return dataset_train_nan, dataset_val_nan, dataset_train, dataset_val, X_normalized, X