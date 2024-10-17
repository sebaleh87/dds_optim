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
    div_term = jnp.exp(jnp.arange(0, embedding_dim, 2) * (-jnp.log(max_position) / embedding_dim))
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

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, in_dict):
        x = in_dict["encoding"]
        for _ in range(self.n_layers - 1):
            x_skip = x
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
            if(_ != 0):
                x = nn.LayerNorm()(x + x_skip)
            else:
                x = nn.LayerNorm()(x)
        
        out_dict = {"embedding": x}
        return out_dict

class EncodingNetwork(nn.Module):
    feature_dim: int = 32
    max_time: float = 1.

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, in_dict):
        x_in = jnp.concatenate([in_dict["x"], in_dict["grads"]], axis=-1)
        t = in_dict["t"]
        t = self.max_time * t
        t_encodings = get_sinusoidal_positional_encoding(t, self.feature_dim, self.max_time)

        x_encode = nn.Dense(self.feature_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x_in)

        x = jnp.concatenate([ x_encode, t, t_encodings], axis=-1)


        return x


class FourierNetwork(nn.Module):
    n_layers: int = 2
    hidden_dim: int = 32
    feature_dim: int = 32
    max_position: float = 10.
    max_time: float = 1.
    mode: str = "continuous"

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x_in, t):
        t = self.max_time * t
        # Concatenate x and t along the last dimension
        t_encodings = FourierFeatureModule(feature_dim=self.hidden_dim)(t)
        x_encodings = FourierFeatureModule(feature_dim=self.hidden_dim)(x_in)
        x = jnp.concatenate([ x_encodings, t_encodings], axis=-1)
        for _ in range(self.n_layers - 1):
            x = FourierFeatureModule(self.hidden_dim)(x)

        return x


class FourierFeatureModule(nn.Module):
    feature_dim: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.feature_dim)(x)
        return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    
#### TODO change and use the code below
from typing import Callable, Optional

class TimeEmbed(nn.Module):
    dim_out: int
    activation: Callable
    num_layers: int = 2
    channels: int = 64
    last_bias_init: Optional[Callable] = None
    last_weight_init: Optional[Callable] = None

    def setup(self):
        # Initialize timestep coefficients
        self.timestep_coeff = jnp.linspace(0.1, 100, self.channels)[None, :]
        self.timestep_phase = self.param(
            'timestep_phase',
            jax.random.normal,
            (1, self.channels)
        )

        # Create hidden layers
        self.hidden_layer = [nn.Dense(self.channels) for _ in range(self.num_layers - 1)]
        self.hidden_layer.insert(0, nn.Dense(self.channels * 2))

        # Output layer with optional bias and weight initialization
        self.out_layer = nn.Dense(
            self.dim_out,
            kernel_init=self.last_weight_init or nn.initializers.lecun_normal(),
            bias_init=self.last_bias_init or nn.initializers.zeros
        )

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        # Ensure t is of shape (batch_size, 1)
        assert t.ndim in [0, 1, 2]
        if t.ndim == 2:
            assert t.shape[1] == 1
        t = t.reshape(-1, 1).astype(jnp.float32)

        # Calculate sinusoidal embeddings
        sin_embed_t = jnp.sin((self.timestep_coeff * t) + self.timestep_phase)
        cos_embed_t = jnp.cos((self.timestep_coeff * t) + self.timestep_phase)

        # Concatenate sinusoidal embeddings
        embed_t = jnp.concatenate([sin_embed_t, cos_embed_t], axis=-1)

        # Pass through hidden layers
        for layer in self.hidden_layer:
            embed_t = self.activation(layer(embed_t))

        # Output layer
        return self.out_layer(embed_t)
