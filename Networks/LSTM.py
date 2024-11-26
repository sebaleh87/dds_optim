import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn







class LSTMNetwork(nn.Module):
    hidden_dim: int = 64
    n_layers: int = 2

    @nn.compact
    def __call__(self, x_dict):

        hidden_state_list = x_dict["hidden_state"]
        encoding = x_dict["encoding"]
        embedding = encoding

        updated_hidden_state_list = []
        for i in range(self.n_layers):
            hidden_state = hidden_state_list[i]
            updated_hidden_state, embedding = nn.OptimizedLSTMCell(self.hidden_dim)(hidden_state, embedding)
            #updated_hidden_state, embedding = nn.LSTMCell(self.hidden_dim)(hidden_state, embedding)
            updated_hidden_state_list.append(updated_hidden_state)

        out_dict = {"embedding": embedding, "hidden_state": updated_hidden_state_list}
        return out_dict