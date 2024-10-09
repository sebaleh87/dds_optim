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
    key = jax.random.PRNGKey(42)

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
