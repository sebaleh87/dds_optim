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
    

class ADAMNetwork(nn.Module):
    
    hidden_dim: int = 64
    n_layers: int = 2

    @nn.compact
    def __call__(self, x_dict):

        prev_momentum = x_dict["hidden_state"][0]
        x_t_encoding = x_dict["encoding"]

        beta_1 = BetaNetwork(self.hidden_dim, self.n_layers)(x_t_encoding)
        beta_2 = BetaNetwork(self.hidden_dim, self.n_layers)(x_t_encoding)

        grad = x_dict["grad"]*MLPNetwork(self.hidden_dim, self.n_layers)(x_t_encoding)
        
        momentum_next = beta_1 * prev_momentum + (1 - beta_1) * grad
        #updated_momentum = MLPNetwork(self.hidden_dim, self.n_layers)(jnp.concatenate(momentum, x_t_encoding, axis = -1))

        prev_velocity = x_dict["hidden_state"][1]
        velocity_next = beta_2 * prev_velocity + (1 - beta_2) * grad**2

        #updated_velocity = MLPNetwork(self.hidden_dim, self.n_layers)(jnp.concatenate(velocity, x_t_encoding, axis = -1))
        
        updated_score = MLPNetwork(self.hidden_dim, self.n_layers)(jnp.concatenate(velocity, momentum, x_t_encoding, axis = -1))#updated_momentum/(jnp.sqrt(updated_velocity) + 1e-8)

        out_dict = {"embedding": updated_score, "hidden_state": (momentum_next, velocity_next)}
        return out_dict
    

class BetaNetwork(nn.Module):
    
    hidden_dim: int = 64
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        
        for _ in range(self.n_layers - 1):
            x_skip = x
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
            if(_ != 0):
                x = nn.LayerNorm()(x + x_skip)
            else:
                x = nn.LayerNorm()(x)

        x = nn.Dense(1, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros)(x)
        x = nn.sigmoid(x)
        
        return x


class MLPNetwork(nn.Module):
    
    hidden_dim: int = 64
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        
        for _ in range(self.n_layers - 1):
            x_skip = x
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
            if(_ != 0):
                x = nn.LayerNorm()(x + x_skip)
            else:
                x = nn.LayerNorm()(x)

        x = nn.Dense(x.shape[-1], kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros)(x)
        
        return x