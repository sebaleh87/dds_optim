
from .LinearSchedule import LinearSchedule
from .ExpSchedule import ExpScheduleClass


AnnealSchedule_registry = {
    "Linear": LinearSchedule,
    "Exp": ExpScheduleClass,
}

def get_AnnealSchedule_class(AnnealConfig):

    return AnnealSchedule_registry[AnnealConfig["name"]](AnnealConfig)