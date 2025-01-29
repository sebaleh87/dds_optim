
from .VanillaBaseModel import VanillaBaseModelClass
from .EquivariantBaseModel import EGNNBaseClass
from .Pisgradnet import PisgradnetBaseClass

BaseModelRegistry = {"Vanilla": VanillaBaseModelClass, "EGNN": EGNNBaseClass, "PISgradnet": PisgradnetBaseClass}


def select_base_network(network_config, SDE_Loss_Config):
    return BaseModelRegistry[network_config["base_name"]](network_config, SDE_Loss_Config)