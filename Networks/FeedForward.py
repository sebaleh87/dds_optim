import flax
from flax import linen as nn
from functools import partial
import jax.numpy as jnp

def get_sinusoidal_positional_encoding(x, embedding_dim, max_position):
    """
    Create a sinusoidal positional encoding as described in the
    "Attention is All You Need" paper.

    Args:
        timestep (int): The current time step.
        embedding_dim (int): The dimensionality of the encoding.

    Returns:
        A 1D tensor of shape (embedding_dim,) representing the
        positional encoding for the given timestep.
    """
    div_term = jnp.exp(np.arange(0, embedding_dim, 2) * (-jnp.log(max_position) / embedding_dim))
    x_pos = jnp.tensordot(x[...,None],div_term[None,:], axes = [[-1],[0]])
    x_sin_embedded = jnp.sin(x_pos)
    x_cos_embedded = jnp.cos(x_pos)

    x_sin_embedded = x_sin_embedded.reshape(x_sin_embedded.shape[:-2] + (-1,))
    x_cos_embedded = x_cos_embedded.reshape(x_cos_embedded.shape[:-2] + (-1,))
    res = jnp.concatenate([x_sin_embedded, x_cos_embedded], axis=-1)
    return res


class FeedForwardNetwork(nn.Module):
    n_layers: int = 2
    hidden_dim: int = 32
    feature_dim: int = 32

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x_in, t):
        # Concatenate x and t along the last dimension
        t = FourierFeatureModule(feature_dim=self.feature_dim)(t)
        x = jnp.concatenate([x_in, t], axis=-1)
        
        for _ in range(self.n_layers - 1):
            x_skip = x
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            if(_ != 0):
                x = nn.LayerNorm()(x + x_skip)
            else:
                x = nn.LayerNorm()(x)
        
        # Output layer without nonlinearity
        x = nn.Dense(x_in.shape[-1])(x)
        return x

class FourierFeatureModule(nn.Module):
    feature_dim: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.feature_dim)(x)
        return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
