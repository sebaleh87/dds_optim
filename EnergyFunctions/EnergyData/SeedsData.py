#https://github.com/tomasgeffner/LDVI/blob/main/src/models/seeds.py

import numpyro
import numpyro.distributions as dist
import jax.numpy as np
import jax
import jax
import numpyro
from jax.flatten_util import ravel_pytree
from .LogisticRegression import load_model_lr

data = {"R": [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3],
		"N": [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7.],
		"X1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		"X2": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
		"tot": 21}

R = np.array(data['R'])
N = np.array(data['N'])
X1 = np.array(data['X1'])
X2 = np.array(data['X2'])
tot = data['tot']

def load_model():
	def model(r):
		tau  = numpyro.sample('tau',  dist.Gamma(0.01, 0.01))
		a_0  = numpyro.sample('a_0',  dist.Normal(0, 10))
		a_1  = numpyro.sample('a_1',  dist.Normal(0, 10))
		a_2  = numpyro.sample('a_2',  dist.Normal(0, 10))
		a_12 = numpyro.sample('a_12', dist.Normal(0, 10))
		with numpyro.plate('J', tot):
			b = numpyro.sample('b', dist.Normal(0, 1 / np.sqrt(tau)))
			logits = a_0 + a_1 * X1 + a_2 * X2 + a_12 * X1 * X2 + b
			r = numpyro.sample('r', dist.BinomialLogits(logits, N), obs = R)
	model_args = (R,)
	return model, model_args

def load_model_other(model_name='Seeds'):
	if model_name == 'Sonar':
		model, model_args = load_model_lr(model_name)
	elif model_name == 'Ionosphere':
		model, model_args = load_model_lr(model_name)
	elif model_name == 'Seeds':
		model, model_args = load_model()
	
	rng_key = jax.random.PRNGKey(1)
	model_param_info, potential_fn, constrain_fn, _ = numpyro.infer.util.initialize_model(rng_key, model, model_args=model_args)
	params_flat, unflattener = ravel_pytree(model_param_info[0])
	log_prob_model = lambda z: -1. * potential_fn(unflattener(z))
	dim = params_flat.shape[0]
	return log_prob_model, dim

if __name__ == "__main__":

	from jax import random
	
	# Initialize random key
	rng_key = random.PRNGKey(0)
	model, model_args = load_model()
	
	# Initialize MCMC
	kernel = numpyro.infer.NUTS(model)
	mcmc = numpyro.infer.MCMC(
		kernel,
		num_warmup=1000,
		num_samples=2000,
		num_chains=1,
	)
	
	# Run MCMC
	mcmc.run(rng_key, R)
	
	# Get samples
	samples = mcmc.get_samples()
	
	# Print summary statistics
	print("\nPosterior Summary:")
	for param in ['tau', 'a_0', 'a_1', 'a_2', 'a_12']:
		mean = np.mean(samples[param])
		std = np.std(samples[param])
		median = np.median(samples[param])
		print(f"{param}: mean = {mean:.3f}, std = {std:.3f}, median = {median:.3f}")
	
	