from .VP_SDE import VP_SDE_Class


SDE_Type_registry = {"VP_SDE": VP_SDE_Class}



def get_SDE_Type_Class(SDE_Type_Config):


    return SDE_Type_registry[SDE_Type_Config["name"]](SDE_Type_Config)

