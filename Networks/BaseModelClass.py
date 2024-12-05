from .FeedForward import FeedForwardNetwork
from .EncodingNetworks import FourierNetwork, EncodingNetwork
from .LSTM import LSTMNetwork
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax
from .VanillaBaseModel import VanillaBaseModelClass
from .EquivariantBaseModel import EGNNBaseClass

NetworkRegistry = {"FeedForward": FeedForwardNetwork, "FourierNetwork": FourierNetwork, "LSTMNetwork": LSTMNetwork}
BaseModelRegistry = {"Vanilla": VanillaBaseModelClass, "EGNN": EGNNBaseClass}

def get_network(network_config, SDE_Loss_Config):
    return NetworkRegistry[network_config["name"]](n_layers=network_config["n_layers"], hidden_dim=network_config["n_hidden"])

def select_base_network(network_config, SDE_Loss_Config):
    return BaseModelRegistry[network_config["name"]](network_config, SDE_Loss_Config)

class BaseModel(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.model = select_base_network(self.network_config, self.SDE_Loss_Config)
        
    @nn.compact
    def __call__(self, in_dict, train = False):
        return self.model.__call__(in_dict, train = train)
        