import numpy as np
import ot
import re
from matplotlib import pyplot as plt
import wandb

def DW_energy( x, dim):
    d = dim
    a = 0
    b = -4
    c = 0.9
    tau = 1.
    """
    Calculate the energy of the Gaussian Mixture Model using logsumexp.
    
    :param x: Input array.
    :return: Energy value.
    """
    d_0 = 4.0
    x = x.reshape(-1, d)
    d_ij = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2 , axis=-1) + 10**-8)
    mask = np.eye(d_ij.shape[0])

    energy_per_particle = a* (d_ij -d_0) + b *(d_ij - d_0)**2 + c*(d_ij - d_0)**4
    energy_per_particle = np.where(mask, 0, energy_per_particle)

    energy = 1/(2*tau) * np.sum(energy_per_particle)
    # energy = jnp.nan_to_num(energy, 10**4)
    # energy = jnp.where(energy > 10**4, 10**4, energy)
    return energy

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

def plot_dm_samples(samples, energy_list, h=3, w=3):
    wandb.init(project="DoubleWell", entity="sanokows")
    plot_samples = samples[:h*w]
    plt.figure(figsize=(8, 8))
    for idx, sample in enumerate(plot_samples):
        plt.subplot(h, w, idx + 1)
        plt.scatter(sample[:, 0], sample[:, 1])
        plt.title(f"Sample {idx}")
    plt.tight_layout()
    wandb.log({"Samples": wandb.Image(plt)})
    plt.show()

    # Plot histogram of interatomic distances
    interatomic_distances = np.sqrt(np.sum((samples[:, None, :, :] - samples[:, :, None, :]) ** 2, axis=-1))
    interatomic_distances = np.ravel(interatomic_distances)
    plt.figure(figsize=(10, 5))
    plt.hist(interatomic_distances.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    plt.title("Histogram of Interatomic Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    wandb.log({"Interatomic Distances Histogram": wandb.Image(plt)})

    # Plot histogram of energy
    plt.figure(figsize=(10, 5))
    plt.hist(energy_list, bins=50, alpha=0.7, color='green', density=True)
    plt.title("Histogram of Energy")
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    wandb.log({"Energy Histogram": wandb.Image(plt)})


    


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

            flattened_com_samples = com_samples.reshape((com_samples.shape[0], -1))
            print("mean, std before reshape", np.mean(flattened_com_samples), np.mean(np.var(flattened_com_samples, axis = 0)))
            if(el == "DW4"):
                energy_list = []
                for com_sample in com_samples:
                    sample = np.ravel(com_sample)
                    energy_list.append(DW_energy(sample, 2))
                #print(energy_list)
                print("DW4 mean energy", np.mean(energy_list), np.std(energy_list))
                print("ad", np.min(energy_list), np.max(energy_list))
                plot_dm_samples(com_samples, energy_list)

            datas.append(split_el)

        # d1 = datas[0]
        # d2 = datas[1]
        # print("computing distance", el)
        # print(wasserstein_distance_w2(d1, d1))
        # print(wasserstein_distance_w2(d1, d2))