import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def generate_gp_samples_with_library(n_groups, n_observations_per_group, n_dimensions, time_points, length_scale=1.0):
    """
    Generate multidimensional GP samples using scikit-learn.
    Each dimension has the same underlying process.
    """
    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8)

    gp_sample_constant = gp.sample_y(time_points.reshape(-1, 1), random_state=None).flatten()

    samples = []
    for _ in range(n_groups):
        group_samples = []
        for _ in tqdm(range(n_observations_per_group)):
            # Generate independent samples for each dimension
            gp_samples_per_dim = []
            for _ in range(n_dimensions):
                gp_sample = gp.sample_y(time_points.reshape(-1, 1), random_state=None).flatten()
                gp_samples_per_dim.append(gp_sample)
            # Stack the samples to form a multidimensional sample
            gp_samples_multidim = np.stack(gp_samples_per_dim, axis=-1)
            group_samples.append(gp_samples_multidim)
        samples.append(np.array(group_samples))

    return samples

def corrupt_by_dim(X):
    """
    Corrupt by taking away some dimensions fully
    """
    X_corr = np.copy(X)
    T, n_dims = X.shape[1], X.shape[2]
    for x in X_corr:
        nb_obs_vals = np.random.randint(0,5)
        obs_mask = np.random.choice(np.arange(T), size = T - nb_obs_vals, replace = False)
        obs_mask = np.isin(np.arange(T), obs_mask)
        n_dims_missing = np.random.randint(1,4)
        dims = np.random.choice(np.arange(n_dims), size = n_dims_missing, replace = False)
        for dim in dims:
            x[obs_mask,dim] = np.nan

    return X_corr

def Generate_samples(n_groups = 1,
                     n_observations_per_group = 5000,
                     n_dimensions = 30,
                     n_time_points = 30,
                     time_points = np.linspace(0, 10, n_time_points),
                     length_scale = 1.5,
                     n_new_dims = 7,
                     p_dataset = .3):

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


    X_normalized = (X_ori - X_ori.mean(axis=(0,1)).reshape(1,1,-1)) / X_ori.std(axis=(0,1)).reshape(1,1,-1)[:100]
    X = corrupt_by_dim(X_normalized.copy())
    X[X!=X] = 0
    dataset = {'X' : X}
    
    len_dataset = len(dataset['X'])
    cut = int(len_dataset*0.7)
    dataset_train, dataset_val = {'X':dataset['X'][:cut]}, {'X':dataset['X'][cut:], 'X_ori':X_normalized[cut:]}

    return X, X_normalized, dataset_train, dataset_val