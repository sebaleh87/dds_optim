
from .GaussianMixture import GaussianMixtureClass
from .MexicanHat import MexicanHatClass
from .Rastrigin import RastriginClass
from .PytheusEnergy import PytheusEnergyClass


Energy_Registry = {"GaussianMixture": GaussianMixtureClass, "MexicanHat": MexicanHatClass, "Rastrigin": RastriginClass, "Pytheus": PytheusEnergyClass}



def get_Energy_class(EnergyConfig):
    """
    Get the Energy class from the Energy Registry.
    
    :param EnergyConfig: Configuration dictionary for the Energy class.
    :return: Energy class.
    """
    return Energy_Registry[EnergyConfig["name"]](EnergyConfig)
