from .BaseModelClass import BaseModel



def get_network(network_config, SDE_Loss_Config):
    return BaseModel(network_config, SDE_Loss_Config)