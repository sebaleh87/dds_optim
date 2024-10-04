from .Base_SDE_Loss import Base_SDE_Loss_Class
from jax import numpy as jnp


class LogVariance_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, EnergyClass, model):
        super().__init__(SDE_config, EnergyClass, model)