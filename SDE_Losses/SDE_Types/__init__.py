from .VP_SDE import VP_SDE_Class
from .VE_SDE import VE_SDE_Class
from .subVP_SDE import subVP_SDE_Class
from .DiscreteTime_SDE import DiscreteTime_SDE_Class

SDE_Type_registry = {"VP_SDE": VP_SDE_Class, "DiscreteTime_SDE": DiscreteTime_SDE_Class, "subVP_SDE": subVP_SDE_Class, "VE_SDE": VE_SDE_Class}



def get_SDE_Type_Class(SDE_Type_Config, Network_Config, Energy_Class):


    return SDE_Type_registry[SDE_Type_Config["name"]](SDE_Type_Config, Network_Config, Energy_Class)

