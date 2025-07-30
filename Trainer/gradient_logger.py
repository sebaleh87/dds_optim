import jax
import jax.numpy as jnp
import optax
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
from typing import List
import pickle


# Flatte and log a single histogram
def log_flat_histogram(tree, name: str, step: int, path = ""):
    eps = 10**-8
    flat_vals, _ = jax.tree_util.tree_flatten(tree)
    all_vals = np.abs(np.concatenate([np.ravel(np.array(v)) for v in flat_vals if v is not None]))

    # Log to wandb
    wandb.log({f"figs/{name}/hist": wandb.Histogram(all_vals)}, step=step)

    # Save and log a matplotlib histogram
    fname = plt.figure()
    plt.hist(all_vals + eps, bins=50, alpha=0.75)
    plt.title(f"{name} histogram at step {step}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xscale('log')
    plt.close()

    wandb.log({f"figs/{name}/img_hist": wandb.Image(fname)}, step=step)

    # Save the flattened values to a pickle file
    log_dir = os.path.join(os.path.dirname(path), "wandb_histograms")
    os.makedirs(log_dir, exist_ok=True)
    pickle_file = os.path.join(log_dir, f"{name}_step_{step}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_vals, f)


