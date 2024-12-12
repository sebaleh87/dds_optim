
from .LinearSchedule import LinearSchedule
from .ExpSchedule import ExpScheduleClass
from .FracSchedule import FracScheduleClass


AnnealSchedule_registry = {
    "Linear": LinearSchedule,
    "Exp": ExpScheduleClass,
    "Frac": FracScheduleClass,
}

def get_AnnealSchedule_class(AnnealConfig):

    return AnnealSchedule_registry[AnnealConfig["name"]](AnnealConfig)