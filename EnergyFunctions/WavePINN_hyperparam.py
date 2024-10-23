from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
import wandb
from matplotlib import pyplot as plt

class WavePINN_hyperparam_Class(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Mexican Hat Potential.
        
        :param a: Position of the minima.
        """
        self.l1_mat_dim = (config["l1_d"], config["d_in"])
        self.l2_mat_dim = (config["l2_d"], config["l2_d"])
        self.l3_mat_dim = (config["d_out"], config["l2_d"])

        self.layer_def = {"l1": self.l1_mat_dim, "l2": self.l2_mat_dim, "l3": self.l3_mat_dim}
        ### TODO inilialize Neural Network here
        self.n_l1_params = self.l1_mat_dim[0] * self.l1_mat_dim[1]
        self.n_l2_params = self.l2_mat_dim[0] * self.l2_mat_dim[1]
        self.n_l3_params = self.l3_mat_dim[0] * self.l3_mat_dim[1]


        self.proj_dim = self.n_l1_params + self.n_l2_params + self.n_l3_params
        self.lam = 2
        super().__init__(config)
        self.proj_dim = self.n_l1_params + self.n_l2_params + self.n_l3_params

    def init_EnergyParams(self):
        jax_key = jax.random.PRNGKey(0)

        param_list = []
        for key in self.layer_def:
            n_params = self.layer_def[key][0]*self.layer_def[key][1]

            jax_key, subkey = jax.random.split(jax_key)
            sampled_params = jax.random.normal(key = subkey, shape = (n_params, self.dim_x))
            nh_target = self.layer_def[key][0] + self.layer_def[key][1]
            sampled_params *= jnp.sqrt(2)/(jnp.sqrt(nh_target)*n_params )
            param_list.append(sampled_params)

        overall_params = jnp.concatenate(param_list, axis = 0) 

        # overall_params = jax.random.normal(key = jax_key, shape = (self.proj_dim, self.dim_x))
        # overall_params *= 1/(self.proj_dim*self.dim_x)
        return {"log_var_x": overall_params}
    
    def scale_samples(self, X, log_sigma):
        sigma = log_sigma["log_var_x"]
        Y = jnp.tensordot(sigma, X, axes = ([-1],[0]))
        return Y
    
    def get_pos(self):
        return jnp.linspace(-1, 1, 200)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, params):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """
        pos = self.get_pos()[..., None]
        pos_dependent_func = lambda pos: self.parameterize_function(params, pos)

        vmap_y = lambda pos: jax.vmap(pos_dependent_func, in_axes=(0,))(pos) 

        grad_y_func = lambda pos: jax.grad(pos_dependent_func)(pos)[...,0]
        grad_grad_y_func = lambda pos: jax.grad(grad_y_func)(pos)[...,0]
        vmap_grad_grad_y_func = lambda pos: jax.vmap(grad_grad_y_func, in_axes=(0))(pos) 

        loss = jnp.mean((vmap_grad_grad_y_func(pos) + self.lam**2 * vmap_y(pos))**2)
        print("loss", loss.shape)
        return loss
    

    def parameterize_function(self, params, x):
        """
        Parameterize the function to be optimized.
        
        :param x: Input array.
        :return: Parameterized function.
        """
        curr_pos = 0
        for idx, layer_defs in enumerate(self.layer_def):
            w = params[curr_pos:curr_pos + self.layer_def[layer_defs][0]*self.layer_def[layer_defs][1]]
            curr_pos += self.layer_def[layer_defs][0]*self.layer_def[layer_defs][1]

            w = w.reshape(self.layer_def[layer_defs])
            x = jnp.tensordot(w, x, axes = ([-1],[0]))
            #print("layer", idx, jnp.mean(x), jnp.std(x))

            if(idx != len(self.layer_def)-1):
                x = nn.relu(x)

        return x[0]
    
    def visualize_samples(self, params):
        """
        Visualize the Mexican Hat Potential.
        """
        params = params[0:10]
        Ys = jax.vmap(self.plot_many_samples, in_axes=(0,))(params)

        fig = plt.figure()
        pos = self.get_pos()[..., None]
        plt.plot(pos, jnp.swapaxes(Ys,0,1))
        wandb.log({"PINN/sampled_solutions": wandb.Image(fig)})
        plt.close()


    def plot_many_samples(self, params):
        pos = self.get_pos()[..., None]

        pos_dependent_func = lambda xx: self.parameterize_function(params, xx)

        vmap_y = lambda xx: jax.vmap(pos_dependent_func, in_axes=(0,))(xx) 

        Ys = vmap_y(pos)
    
        return Ys

