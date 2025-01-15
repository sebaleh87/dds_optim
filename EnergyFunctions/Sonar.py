from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import pandas as pd
import os
import pickle

class SonarClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Funnel distribution (Neal, 2003).
        The distribution is defined as:
        x1 ~ N(0, η²)
        xi ~ N(0, exp(x1)) for i = 2,...,dim
        
        :param config:
        """
        super().__init__(config)

        self.dim = 60

        self.sonar_data, self.sonar_labels = self.load_sonar_dataset()

    def load_sonar_dataset(self):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(f'{current_folder}/EnergyData/sonar/sonar_full.pkl', 'rb') as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2
        return X, Y


    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """

        log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=0, scale=1), axis = -1)
        # print(x)
        # print(self.sonar_data)
        # print("here",jax.nn.sigmoid(jnp.dot(self.sonar_data, x)))

        # log_p_0 = jnp.log()
        # log_bernoulli = jnp.where(self.sonar_label == 0, , )
        eps = 1e-6
        sigmoid = jnp.clip(jax.nn.sigmoid(jnp.sum(self.sonar_data* x[None, :], axis = -1)), min = eps, max = 1-eps)
        # sigmoid = jnp.where((self.sonar_labels == 0) *(1 - sigmoid < eps) , 1 - sigmoid + eps, 1-sigmoid)
        # sigmoid = jnp.where((self.sonar_labels == 1) *(sigmoid < eps) , sigmoid + eps, sigmoid)
        log_bernoulli = jnp.sum(jax.scipy.stats.bernoulli.logpmf(self.sonar_labels, sigmoid), axis = -1)

        energy = -log_prior - log_bernoulli
        return energy

    