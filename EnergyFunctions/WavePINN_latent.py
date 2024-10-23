from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
import wandb
from matplotlib import pyplot as plt

class FeedForwardNetwork(nn.Module):
    n_layers: int = 2
    hidden_dim: int = 32
    n_out: int = 1

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x, z):
        x = jnp.concatenate([x, z], axis = -1)
        for _ in range(self.n_layers - 1):
            x_skip = x
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
            if(_ != 0):
                x = nn.LayerNorm()(x + x_skip)
            else:
                x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.n_out, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
        return x


class WavePINN_latent_Class(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Mexican Hat Potential.
        
        :param a: Position of the minima.
        """
        self.lam = 2.
        self.pos_batch_size = 200
        super().__init__(config)
        self.model = FeedForwardNetwork( n_layers = 3, hidden_dim = 64, n_out = 1)
        self.pos_dim = 1
        self.max_val = 2*jnp.pi
        self.min_val = -2*jnp.pi

    def init_EnergyParams(self):
        params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.dim_x)), jnp.ones((1, self.pos_dim)))
        return params
    
    def scale_samples(self, X, network_params, key):
        key, subkey = jax.random.split(key)
        pos = self.get_pos(subkey)

        Y_func = lambda x_pos: self.parameterize_function(network_params, X, x_pos)
        vmap_Y = jax.vmap(Y_func, in_axes=(0,))( pos)

        grad_Y = lambda x_pos: jax.grad(Y_func)(x_pos)[0]
        grad_grad_Y = lambda x_pos: jax.grad(grad_Y)(x_pos)

        vmap_Y = jax.vmap(Y_func, in_axes=(0,))( pos)
        vmap_Y_grad_grad = jax.vmap(grad_grad_Y, in_axes=(0,))(pos)

        Y = jnp.concatenate([vmap_Y[...,None], vmap_Y_grad_grad, pos], axis = -1)
        #print("shapes", Y.shape, vmap_Y.shape, vmap_Y_grad_grad.shape)
        return Y, key
    
    def get_pos(self, subkey):
        pos = jax.random.uniform(subkey, (self.pos_batch_size, self.pos_dim), minval = self.min_val, maxval = self.max_val)
        return pos
    
    def get_deterministic_pos(self):
        return jnp.linspace(self.min_val, self.max_val, self.pos_batch_size)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, Y):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """
        f = Y[...,0]
        f_grad_grad = Y[..., 1]
        loss = jnp.mean((self.lam**2*f + f_grad_grad)**2)
        return loss


    def parameterize_function(self, params, diff_samples, pos):
        """
        Parameterize the function to be optimized.
        
        :param x: Input array.
        :return: Parameterized function.
        """
        x = self.model.apply(params, diff_samples, pos)

        return x[0]
    
    def visualize_samples(self, Ys):
        """
        Visualize the Mexican Hat Potential.
        """
        Ys = jnp.swapaxes(Ys, 0, 1)
        fig = plt.figure()
        plt.plot(Ys[...,-1], Ys[...,0], "x")
        wandb.log({"PINN/sampled_solutions": wandb.Image(fig)})
        plt.close()


    def plot_many_samples(self, diff_samples, energy_params):
        pos = self.get_deterministic_pos()[..., None]

        pos_dependent_func = lambda xx: self.parameterize_function(energy_params, diff_samples, xx)

        vmap_y = lambda xx: jax.vmap(pos_dependent_func, in_axes=(0,))(xx) 

        Ys = vmap_y(pos)
    
        return Ys

