
from .GaussianMixture import GaussianMixtureClass
from .MexicanHat import MexicanHatClass
from .Rastrigin import RastriginClass
from .PytheusEnergy import PytheusEnergyClass
from .WavePINN_hyperparam import WavePINN_hyperparam_Class
from .WavePINN_latent import WavePINN_latent_Class
from .LennardJones import LennardJonesClass
from .DoubleWell import DoubleWellClass
from .DoubleMoon import DoubleMoonClass


Energy_Registry = {"GaussianMixture": GaussianMixtureClass, "MexicanHat": MexicanHatClass, "Rastrigin": RastriginClass, "Pytheus": PytheusEnergyClass,
                "WavePINN_hyperparam": WavePINN_hyperparam_Class, "WavePINN_latent": WavePINN_latent_Class, "LennardJones": LennardJonesClass,
                "DoubleWell_iter": DoubleWellClass, "DoubleWell_Richter": DoubleWellClass, "DoubleMoon": DoubleMoonClass}



def get_Energy_class(EnergyConfig):
    """
    Get the Energy class from the Energy Registry.
    
    :param EnergyConfig: Configuration dictionary for the Energy class.
    :return: Energy class.
    """
    return Energy_Registry[EnergyConfig["name"]](EnergyConfig)
