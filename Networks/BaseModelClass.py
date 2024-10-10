from .FeedForward import FeedForwardNetwork, FourierNetwork
from flax import linen as nn
from functools import partial
import jax.numpy as jnp


NetworkRegistry = {"FeedForward": FeedForwardNetwork, "FourierNetwork": FourierNetwork}


def get_network(network_config, SDE_Loss_Config):
    return NetworkRegistry[network_config["name"]](n_layers=network_config["n_layers"], hidden_dim=network_config["n_hidden"], feature_dim=network_config["feature_dim"], max_time = SDE_Loss_Config["n_integration_steps"])

class BaseModel(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.backbone = get_network(self.network_config, self.SDE_Loss_Config)
        
    @nn.compact
    def __call__(self, x_in, t):
        embedding = self.backbone(x_in, t)

        x_dim = x_in.shape[-1]
        if(self.SDE_mode == "DiscreteTime_SDE"):
            mean_x = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            log_var_x = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                  bias_init=nn.initializers.zeros)(embedding)

            return mean_x, log_var_x
        else:
            x = nn.Dense(x_dim)(embedding)
            x = jnp.clip(x, -10**4, 10**4)
            return x