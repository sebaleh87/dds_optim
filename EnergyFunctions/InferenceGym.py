from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import pandas as pd
import inference_gym.using_jax as gym

class InferenceGymClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Funnel distribution (Neal, 2003).
        The distribution is defined as:
        x1 ~ N(0, η²)
        xi ~ N(0, exp(x1)) for i = 2,...,dim
        
        :param config:
        """
        super().__init__(config)

        self.name = config["name"]
        self.log_prob_model, self.dim = self.load_model_gym(self.name)

    def load_model_gym(self, model='banana'):
        def log_prob_model(z):
            x = target.default_event_space_bijector(z)
            return (target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims = 1))
        if model == 'Lorenz':
            target = gym.targets.ConvectionLorenzBridge()
        if model == 'Brownian':
            target = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
        if model == 'Banana':
            target = gym.targets.Banana()
        target = gym.targets.VectorModel(target, flatten_sample_transformations=True)
        dim = target.event_shape[0]
        return log_prob_model, dim


    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """

        return -self.log_prob_model(x)
    