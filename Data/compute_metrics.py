import numpy as np
import ot
import re

def wasserstein_distance_w2(samples1, samples2, weights1=None, weights2=None):
    """
    Compute the W2 Wasserstein distance between two discrete distributions.
    
    Args:
        samples1 (ndarray): Array of shape (n1, d) representing samples from the first distribution.
        samples2 (ndarray): Array of shape (n2, d) representing samples from the second distribution.
        weights1 (ndarray): Array of shape (n1,) representing weights for samples1. Defaults to uniform weights.
        weights2 (ndarray): Array of shape (n2,) representing weights for samples2. Defaults to uniform weights.
        
    Returns:
        float: The W2 Wasserstein distance between the two distributions.
    """
    n1, n2 = samples1.shape[0], samples2.shape[0]
    
    # Compute pairwise squared Euclidean distances
    cost_matrix = np.linalg.norm(samples1[:, None, :] - samples2[None, :, :], axis=2)**2
    
    # Use uniform weights if not provided
    if weights1 is None:
        weights1 = np.ones(n1) / n1
    if weights2 is None:
        weights2 = np.ones(n2) / n2
    
    # Compute optimal transport plan using the Hungarian algorithm
    transport_plan = ot.emd(weights1, weights2, cost_matrix)
    
    # Compute the Wasserstein distance
    wasserstein_distance = np.sum(transport_plan * cost_matrix)
    
    return np.sqrt(wasserstein_distance)

# Example usage
if __name__ == "__main__":
    # Generate random samples from two distributions
    np.random.seed(42)
    samples1 = np.random.randn(100, 2)  # 100 samples from a 2D Gaussian
    samples2 = np.random.randn(120, 2) + 1  # 120 samples from a shifted 2D Gaussian
    
    # Compute the W2 Wasserstein distance
    w2_distance = wasserstein_distance_w2(samples1, samples2)
    print(f"W2 Wasserstein distance: {w2_distance}")

    # Load the test_split_DW4.npy file
    split = [ "val", "test"]
    datasets = ["DW4", "LJ13-1000", "LJ55-1000-part1"]
    for el in datasets:

        match = re.search(r'\d+', el)
        if match:
            n_paricles = int(match.group())
        else:
            raise ValueError(f"No number found in string: {el}")
        datas = []
        for s in split:
            split_el = np.load(f'/system/user/publicwork/sanokows/Denoising_diff_sampler/Data/{s}_split_{el}.npy')
            print(f"Loaded {s} with shape: {split_el.shape}")

            samples = split_el.reshape((split_el.shape[0], n_paricles, -1))
            #print(samples)
            #print("mean", np.mean(samples, axis = 0), np.var(samples, axis = 0), np.mean(split_el[:,0:-1:2], axis = 0))

            com_samples = samples - np.mean(samples, axis = 1, keepdims=True)
            #print(samples)
            print(el, "com mean", np.mean(np.mean(com_samples, axis = 0), axis = 0), np.mean(np.var(com_samples, axis = 0), axis = 0))

            datas.append(split_el)

        # d1 = datas[0]
        # d2 = datas[1]
        # print("computing distance", el)
        # print(wasserstein_distance_w2(d1, d1))
        # print(wasserstein_distance_w2(d1, d2))