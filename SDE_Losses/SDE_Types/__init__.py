from .VP_SDE import VP_SDE_Class
from .DiscreteTime_SDE import DiscreteTime_SDE_Class

SDE_Type_registry = {"VP_SDE": VP_SDE_Class, "DiscreteTime_SDE": DiscreteTime_SDE_Class}



def get_SDE_Type_Class(SDE_Type_Config):


    return SDE_Type_registry[SDE_Type_Config["name"]](SDE_Type_Config)

