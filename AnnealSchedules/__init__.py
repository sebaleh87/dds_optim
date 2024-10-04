
from .LinearSchedule import LinearSchedule


AnnealSchedule_registry = {
    "Linear": LinearSchedule
}

def get_AnnealSchedule_class(AnnealConfig):

    return AnnealSchedule_registry[AnnealConfig["name"]](AnnealConfig)