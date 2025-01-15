from jax import grad, vmap
import jax.numpy as np
import inference_gym.using_jax as gym
import jax.numpy as jnp

def load_model_gym(model='banana'):
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

if(__name__ == "__main__"):
    # Load the model
    log_prob_model, dim = load_model_gym(model='brownian')
    log_probs = log_prob_model(jnp.ones((dim, )))
    print(log_probs)
    # Set the random seed