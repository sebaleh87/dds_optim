from .GaussianMixture import GaussianMixtureClass
from .MexicanHat import MexicanHatClass
from .Rastrigin import RastriginClass
from .PytheusEnergy import PytheusEnergyClass
from .WavePINN_hyperparam import WavePINN_hyperparam_Class
from .WavePINN_latent import WavePINN_latent_Class
from .LennardJones import LennardJonesClass
from .DoubleWell import DoubleWellClass
from .DoubleWellEquivariant import DoubleWellEquivariantClass
from .MW54 import ManyWellClass
from .MW54_1 import ManyWellClass1
from .DoubleMoon import DoubleMoonClass
from .Sonar import SonarClass
from .InferenceGym import InferenceGymClass
from .Funnel import FunnelClass
from .LGCP import LGCPClass
from .GermanCredit import GermanCreditClass
from .StudentTMixture import StudentTMixtureClass
from .GMMdistrax import GMMDistraxClass
#from .FunnelDistrax import FunnelDistraxClass

Energy_Registry = {"GaussianMixture": GaussianMixtureClass, "MexicanHat": MexicanHatClass, "Rastrigin": RastriginClass, "Pytheus": PytheusEnergyClass,
                "WavePINN_hyperparam": WavePINN_hyperparam_Class, "WavePINN_latent": WavePINN_latent_Class, "LennardJones": LennardJonesClass,
                "DoubleWellEquivariant": DoubleWellEquivariantClass, "DoubleWell": DoubleWellClass, "DoubleMoon": DoubleMoonClass,
                "Lorenz": InferenceGymClass, "Brownian": InferenceGymClass, "Banana": InferenceGymClass, "Seeds": InferenceGymClass, 
                "Ionosphere" : InferenceGymClass, "Sonar": InferenceGymClass, "Funnel": FunnelClass, "LGCP": LGCPClass, "GermanCredit": GermanCreditClass,
                "MW54": ManyWellClass,
                "MW54_1": ManyWellClass1,
                "StudentTMixture": StudentTMixtureClass, "GMMDistrax": GMMDistraxClass
                #"FunnelDistrax": FunnelDistraxClass
                }


def get_Energy_class(EnergyConfig):
    """
    Get the Energy class from the Energy Registry.
    
    :param EnergyConfig: Configuration dictionary for the Energy class.
    :return: Energy class.
    """
    return Energy_Registry[EnergyConfig["name"]](EnergyConfig)
