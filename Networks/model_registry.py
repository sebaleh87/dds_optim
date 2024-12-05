from .FeedForward import FeedForwardNetwork
from .LSTM import LSTMNetwork

NetworkRegistry = {"FeedForward": FeedForwardNetwork, "LSTMNetwork": LSTMNetwork}


def get_network(network_config, SDE_Loss_Config):
    return NetworkRegistry[network_config["name"]](n_layers=network_config["n_layers"], hidden_dim=network_config["n_hidden"])

