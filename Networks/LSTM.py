import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn







class LSTM(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x_dict):

        hidden_state = x_dict["hidden_state"]
        encoding = x_dict["encoding"]

        updated_hidden_state, embedding = nn.OptimizedLSTMCell(hidden_state, encoding)

        out_dict = {"x_out": embedding, "hidden_state": updated_hidden_state}
        return out_dict