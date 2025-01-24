import numpy as np

def sample_and_remove(A, t, T):
    # Create a new dictionary where the keys are the same as A,
    # but the values are sampled t elements from each list in A.
    indices = np.random.choice(T, t, replace=False).tolist()
    sampled_dict = {k:  [v[el] for el in indices] for k, v in A.items()}

    # Remove the sampled elements from the original lists in A
    A = {k: [item for item in v if item not in sampled_dict[k]] for k, v in A.items()}

    return sampled_dict, A


import jax
import jax.numpy as jnp
import numpy as np

def sample_and_remove_entries(data_dict, t, key):
    """
    Randomly sample t*B indices from the input dictionary and save them into a new dictionary.
    Then, remove the sampled entries from the original dictionary.
    
    Args:
    - data_dict: a dictionary where each key is an array of shape (T, B, 1) or (T).
    - t: number of rows to sample.
    - key: jax.random.PRNGKey for generating random indices.
    
    Returns:
    - sampled_dict: a dictionary with sampled entries.
    - remaining_dict: a dictionary with remaining entries after sampling.
    """
    sampled_dict = {}
    remaining_dict = {}
    
    # Get the first key to extract the dimensions (assuming all arrays have the same shape)
    first_key = list(data_dict.keys())[0]
    T = data_dict[first_key].shape[0]
    B = data_dict[first_key].shape[1]
    
    # Generate t*B unique random indices in range T
    Ts =  jnp.repeat(np.arange(0,T)[:, None], B, axis = -1)
    random_indices = jax.random.choice(key, Ts, shape=(t,B), replace=False)

    for k, array in data_dict.items():
        if array.ndim == 1:
            remaining_dict[k] = jnp.repeat(array[:, None], B, axis = -1)
        else:
            remaining_dict[k] = array

    for k, array in data_dict.items():
        B = array.shape[1]
        # Sample corresponding (t, B, 1) entries
        sampled_array = array[random_indices[...,None], :, :]
        # Remaining entries
        remaining_array = jnp.delete(array, random_indices[...,None], axis=0)
        sampled_dict[k] = sampled_array
    
    return sampled_dict, remaining_dict

if(__name__ == "__main__"):
    # Example usage:
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(42)

    bs = 10
    x_dim = 3
    a = jnp.ones((bs,x_dim))
    b = jnp.ones((x_dim))
    c = jnp.ones((bs,))
    

    res1 =  a*b
    res2 = b*a
    res3 = b+a
    #res5 = b+c

    print(res1.shape, res2.shape, res3.shape)

    import numpy as np

    # Parameters for q(x) and q(y|x)
    lambda_q = 1.0  # Rate parameter for q(x)
    delta_q = 0.5  # Width for uniform q(y|x)

    # Parameters for p(x) and p(y|x)
    lambda_p = 2.0  # Rate parameter for p(x)
    delta_p = 0.3  # Width for uniform p(y|x)

    # Number of samples
    n_samples = 100000

    # Sample from q(x)
    x_samples = np.random.exponential(1 / lambda_q, n_samples)

    # Sample from q(y|x)
    y_samples = np.random.uniform(x_samples - delta_q, x_samples + delta_q, n_samples)

    # Compute log density ratios
    # log(q(x)/p(x)) for exponential distributions
    log_q_p_x = np.log(lambda_q / lambda_p) + (lambda_p - lambda_q) * x_samples

    # log(q(y|x)/p(y|x)) for uniform distributions
    q_y_given_x = 1 / (2 * delta_q)
    p_y_given_x = np.where(
        (y_samples >= x_samples - delta_p) & (y_samples <= x_samples + delta_p),
        1 / (2 * delta_p),
        0,
    )
    log_q_p_y_given_x = np.log(q_y_given_x) - np.log(p_y_given_x, out=np.zeros_like(p_y_given_x), where=p_y_given_x > 0)

    # Handle cases where p_y_given_x = 0 (log(0) -> -inf)
    log_q_p_y_given_x[p_y_given_x == 0] = -np.inf

    # Compute variance and covariance
    var_log_q_p_x = np.var(log_q_p_x)
    covar = np.cov(log_q_p_x, log_q_p_y_given_x, rowvar=False)[0, 1]

    # Check the condition
    condition = var_log_q_p_x + covar < 0

    # Output results
    print(f"Variance of log(q(x)/p(x)): {var_log_q_p_x:.4f}")
    print(f"Covariance of log(q(x)/p(x)) and log(q(y|x)/p(y|x)): {covar:.4f}")
    print(f"Condition met (variance + covariance < 0): {condition}")
    raise ValueError("Stop here")


    import numpy as np
    rhos = 10*np.linspace(-1, 1, 100)
    for rho in rhos:
        # Parameters for q(x) and q(y|x)
        mu_x = 0.5
        sigma_x = 1.5
        rho = rho # Correlation between x and y
        sigma_y = 1.0

        # Parameters for p(x) and p(y|x)
        mu_p = 0.0
        sigma_p_x = 1.0
        sigma_p_y = 1.0

        # Number of samples
        n_samples = 100000

        # Sample from q(x)
        x_samples = np.random.normal(mu_x, sigma_x, n_samples)

        # Sample from q(y|x)
        y_samples = rho * x_samples + np.random.normal(0, sigma_y, n_samples)

        # Define the log ratios
        log_q_p_x = -0.5 * (((x_samples - mu_x) / sigma_x) ** 2 - ((x_samples - mu_p) / sigma_p_x) ** 2)
        log_q_p_y_given_x = -0.5 * (
            ((y_samples - rho * x_samples) / sigma_y) ** 2
            - ((y_samples - x_samples) / sigma_p_y) ** 2
        )

        # Compute variance and covariance
        var_log_q_p_x = np.var(log_q_p_x)
        covar = np.cov(log_q_p_x, log_q_p_y_given_x)[0, 1]

        # Check the condition
        condition = var_log_q_p_x + covar < 0

        # Output results
        print(rho)
        print(f"Variance of log(q(x)/p(x)): {var_log_q_p_x:.4f}")
        print(f"Covariance of log(q(x)/p(x)) and log(q(y|x)/p(y|x)): {covar:.4f}")
        print(f"Condition met (variance + covariance < 0): {condition}")

    raise ValueError("Stop here")
    import matplotlib.pyplot as plt

    # Generate sigma_scale_factor
    subkey, _ = jax.random.split(key)
    shape = (100000,)
    scale_strength = 0.01
    sigma_scale_factor = 1 + jax.random.exponential(subkey, shape) * scale_strength

    # Plot histogram of sigma_scale_factor
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(sigma_scale_factor, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of sigma_scale_factor')
    plt.xlabel('sigma_scale_factor')
    plt.ylabel('Density')

    # Plot log probability of sigma_scale_factor
    log_prob = jnp.log(sigma_scale_factor)
    plt.subplot(1, 2, 2)
    plt.hist(log_prob, bins=30, density=True, alpha=0.6, color='b')
    plt.title('Log Probability of sigma_scale_factor')
    plt.xlabel('log(sigma_scale_factor)')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()
    import os
    import jax
    # Save the figure to /Figures
    print(os.getcwd())
    plt.savefig(os.getcwd() + '/Figures/sigma_scale_factor_histogram.png')

    raise ValueError("Stop here")
    sigma_scale_factor = 1 + jax.random.exponential(subkey, shape) * scale_strength

    # Example dictionary with arrays of shape (T, B, 1) and (T)
    data_dict = {
        'a': jnp.ones((10, 3, 1)),
        'b': jnp.arange(10),
    }

    # Sample 5 indices from (T, B, 1) and (T) arrays
    sampled_dict, remaining_dict = sample_and_remove_entries(data_dict, t=5, key=key)

    print("Sampled dict:")
    print(sampled_dict)
    print("\nRemaining dict:")
    print(remaining_dict)
