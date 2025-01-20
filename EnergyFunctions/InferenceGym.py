from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import inference_gym.using_jax as gym
import numpyro
from jax.flatten_util import ravel_pytree
from .EnergyData import SeedsData as model_seeds
from .EnergyData.SeedsData import load_model_other

class InferenceGymClass(EnergyModelClass):
    def __init__(self, config):
        super().__init__(config)


        self.name = config["name"]
        
        # Handle different model types
        if self.name in ['Sonar', 'Ionosphere', 'Seeds']:
            self.log_prob_model, self.dim_x = self.load_model_numpyro(self.name)
        else:
            self.log_prob_model, self.dim_x = self.load_model_gym(self.name)

        super().__init__(config)

    def load_model_gym(self, model='banana'):
        def log_prob_model(z):
            x = target.default_event_space_bijector(z)
            # print(x)
            # print(z)
            # print(target.unnormalized_log_prob(x) )
            # print(target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims = 1))
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

    def load_model_numpyro(self, model='Seeds'):
        return load_model_other(model)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """
        if(self.name == "Brownian"):
            min_val = -10
            #clipped_x = jnp.clip(x[0:2], min = -10)
            clipped_x = jnp.where(x[0:2] < min_val, jax.nn.leaky_relu(x[0:2] - min_val), x[0:2])
            x = x.at[0:2].set(clipped_x)

        return -self.log_prob_model(x)
    
if __name__ == "__main__":
    model = InferenceGymClass(config = {'name': 'Seeds', 'dim_x': 26, 'scaling': 1.0})
    print(model.energy_function(jnp.zeros(26)))

