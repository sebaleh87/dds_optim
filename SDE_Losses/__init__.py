from .Base_SDE_Loss import Base_SDE_Loss_Class
from .Reverse_KL_Loss import Reverse_KL_Loss_Class
from .LogVariance_Loss import LogVariance_Loss_Class


SDE_Loss_registry = {"Reverse_KL_Loss": Reverse_KL_Loss_Class, "LogVariance_Loss": LogVariance_Loss_Class}

def get_SDE_Loss_class(SDE_Loss_Config, Energy_Class, model):
    return SDE_Loss_registry[SDE_Loss_Config["name"]](SDE_Loss_Config, Energy_Class, model)