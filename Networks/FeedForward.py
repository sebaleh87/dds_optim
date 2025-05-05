import flax
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax


class FeedForwardNetwork(nn.Module):
    n_layers: int = 3
    hidden_dim: int = 64

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, in_dict):
        x = in_dict["encoding"]
        for _ in range(self.n_layers):
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
