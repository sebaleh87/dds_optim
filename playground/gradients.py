import jax
import jax.numpy as jnp
import optax
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
from typing import List

# Initialize wandb
wandb.init(project="jax-mlp-histograms", name="mlp-run", mode="online")

# Simple dataset: regression target is sum of inputs
def get_data(batch_size=64, input_dim=10):
    X = np.random.randn(batch_size, input_dim)
    y = np.sum(X, axis=1, keepdims=True)
    return X, y

# Initialize MLP parameters
def init_mlp_params(key, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int = 1):
    keys = jax.random.split(key, n_layers + 1)
    params = []
    in_dim = input_dim
    for i in range(n_layers):
        w = jax.random.normal(keys[i], (in_dim, hidden_dim)) * jnp.sqrt(2. / in_dim)
        b = jnp.zeros((hidden_dim,))
        params.append({"w": w, "b": b})
        in_dim = hidden_dim
    # Output layer
    w_out = jax.random.normal(keys[-1], (in_dim, output_dim)) * jnp.sqrt(2. / in_dim)
    b_out = jnp.zeros((output_dim,))
    params.append({"w": w_out, "b": b_out})
    return params

# MLP forward pass
def mlp_forward(params: List[dict], x):
    for layer in params[:-1]:
        x = jnp.dot(x, layer["w"]) + layer["b"]
        x = jax.nn.relu(x)
    out_layer = params[-1]
    return jnp.dot(x, out_layer["w"]) + out_layer["b"]

# Loss function (MSE)
def loss_fn(params, x, y):
    preds = mlp_forward(params, x)
    return jnp.mean((preds - y) ** 2)

# Flatten and log a single histogram
def log_flat_histogram(tree, name: str, step: int):
    flat_vals, _ = jax.tree_util.tree_flatten(tree)
    all_vals = np.concatenate([np.ravel(np.array(v)) for v in flat_vals if v is not None])

    # Log to wandb
    wandb.log({f"figs/{name}/hist": wandb.Histogram(all_vals)}, step=step)

    # Save and log a matplotlib histogram
    fig = plt.figure()
    plt.hist(all_vals, bins=50, alpha=0.75)
    plt.title(f"{name} histogram at step {step}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    file_path = os.path.realpath(__file__)
    fname = file_path + f"hist_{name}_step_{step}.png"
    plt.savefig(fname)
    plt.close()
    wandb.log({ f"figs/{name}/fig": wandb.Image(fig)}, step=step)

# Training step
def train_step(params, opt_state, x, y, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, grads, updates, new_params, new_opt_state

# Main training loop
def train(num_steps=200, input_dim=10, hidden_dim=64, n_layers=3):
    key = jax.random.PRNGKey(42)
    params = init_mlp_params(key, input_dim, hidden_dim, n_layers)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    for step in range(num_steps):
        x_batch, y_batch = get_data(input_dim=input_dim)
        x_batch = jnp.array(x_batch)
        y_batch = jnp.array(y_batch)

        loss, grads, updates, params, opt_state = train_step(params, opt_state, x_batch, y_batch, optimizer)
        wandb.log({"loss": loss}, step=step)

        if step % 10 == 0:
            log_flat_histogram(grads, "grads", step)
            log_flat_histogram(updates, "updates", step)

if __name__ == "__main__":
    os.makedirs("wandb_histograms", exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
                           # GPU -1 means select GPU via env var in command line
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{3}"
    train(num_steps=200, input_dim=10, hidden_dim=64, n_layers=10)
