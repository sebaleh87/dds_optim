import jax
import jax.numpy as jnp
import numpy as np
import os
from .BaseEnergy import EnergyModelClass

class GermanCreditClass(EnergyModelClass):
    def __init__(self, EnergyConfig):
        super().__init__(EnergyConfig)
        self.dim_x = 25  # Fixed dimension for German Credit data
        
        # Load the numeric dataset directly
        dataset_path = "EnergyFunctions/EnergyData/Datasets/german.data-numeric"
        data = np.loadtxt(dataset_path)
        
        # Prepare features and labels exactly as in reference
        X = data[:, :-1]  # Keep as float64 initially
        X /= jnp.std(X, 0)[jnp.newaxis, :]  # Use jax's std
        X = jnp.hstack((jnp.ones((len(X), 1)), X))
        self.data = jnp.array(X, dtype=jnp.float32)  # Convert to float32 after preprocessing
        
        # Process labels exactly as in reference
        self.labels = data[:, -1] - 1
        self.labels = jnp.array(jnp.expand_dims(self.labels.astype(jnp.float32), 1))
        

    def energy_function(self, x):
        def _log_prob(x):
            features = -jnp.matmul(self.data, x.transpose())
            log_likelihood = jnp.sum(
                jnp.where(
                    self.labels == 1,
                    jax.nn.log_sigmoid(features),
                    jax.nn.log_sigmoid(features) - features
                ),
                axis=0
            )
            return log_likelihood

        # Handle batching exactly as in reference
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
            
        log_probs = _log_prob(x)
        
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)
            
        # Return negative log probability as energy
        return -log_probs






