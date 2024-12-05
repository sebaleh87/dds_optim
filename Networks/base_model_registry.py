
from .VanillaBaseModel import VanillaBaseModelClass
from .EquivariantBaseModel import EGNNBaseClass

BaseModelRegistry = {"Vanilla": VanillaBaseModelClass, "EGNN": EGNNBaseClass}


def select_base_network(network_config, SDE_Loss_Config):
    return BaseModelRegistry[network_config["base_name"]](network_config, SDE_Loss_Config)