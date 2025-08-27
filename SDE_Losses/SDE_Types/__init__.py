from .VP_SDE import VP_SDE_Class
from .VE_SDE import VE_SDE_Class
from .Bridge_SDE import Bridge_SDE_Class
from .VE_Discrete import VE_Discrete_Class

SDE_Type_registry = {"VP_SDE": VP_SDE_Class, "VE_Discrete": VE_Discrete_Class, "VE_SDE": VE_SDE_Class, "Bridge_SDE": Bridge_SDE_Class}



def get_SDE_Type_Class(SDE_Type_Config, Network_Config, Energy_Class):

    return SDE_Type_registry[SDE_Type_Config["name"]](SDE_Type_Config, Network_Config, Energy_Class)

