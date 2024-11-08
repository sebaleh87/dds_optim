import optax
import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
import wandb
from matplotlib import pyplot as plt

class FeedForwardNetwork(nn.Module):
    n_layers: int = 3
    hidden_dim: int = 32
    n_out: int = 1

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x, z):
        x = jnp.concatenate([x, z], axis = -1)
        for _ in range(self.n_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            x = nn.tanh(x)

        
        x = nn.Dense(self.n_out, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros)(x)
        return x


class WavePINN_latent_Class():
    def __init__(self, config):
        """
        Initialize the Mexican Hat Potential.
        
        :param a: Position of the minima.
        """
        self.lam = 1
        self.pos_batch_size = 1000
        self.config = config
        self.lr_factor = 1.
        self.model = FeedForwardNetwork( n_layers = 3, hidden_dim = 40, n_out = 1)
        self.pos_dim = 1
        self.dim_x = 1	
        self.max_val = 2*jnp.pi
        self.min_val = 0

        self.init_wandb(config["project_name"], config)
    
    def init_wandb(self, project_name, config):
        """
        Initialize Weights and Biases (wandb) for logging.
        
        :param project_name: Name of the wandb project.
        :param config: Configuration dictionary for wandb.
        """
        wandb.init(project=project_name, config=config)

    def init_EnergyParams(self):
        params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.dim_x)), jnp.ones((1, self.pos_dim)))
        return params

    @partial(jax.jit, static_argnums=(0,2))
    def update_params(self, grads, optimizer, params, state):
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state

    def train(self):
        epochs = self.config["epochs"]
        batch_size = 5
        self.Energy_params = self.init_EnergyParams()
        optimizer = self.init_Energy_params_optimizer()#optax.adam(lr)
        self.Energy_params_state = optimizer.init(self.Energy_params)
        self.Energy_key = jax.random.PRNGKey(0)


        for epoch in range(epochs):
            self.Energy_key, subkey = jax.random.split(self.Energy_key)
            X = 0.1*jax.random.normal(subkey, (batch_size,self.dim_x))

            (loss, self.Energy_key), grads = jax.value_and_grad(self.vmap_calc_energy, argnums = 1, has_aux=True)(X, self.Energy_params, self.Energy_key)

            self.Energy_params, self.Energy_params_state = self.update_params(grads, optimizer, self.Energy_params, self.Energy_params_state)

            print("Epoch: ", epoch, "Loss: ", loss)
            wandb.log({"PINN/loss": loss})

            if(epoch % 100 == 0):
                self.Energy_key, subkey = jax.random.split(self.Energy_key)
                X = jax.random.normal(subkey, (batch_size,self.dim_x))
                self.Energy_key, subkey = jax.random.split(self.Energy_key)
                batched_key = jax.random.split(subkey, batch_size)
                Ys, _ = jax.vmap(self.scale_samples, in_axes=(0, None, 0))(X, self.Energy_params, batched_key)
                self.visualize_samples(Ys)


    def init_Energy_params_optimizer(self):
        l_start = 1e-10
        l_max = self.config["lr"]
        lr_min = l_max/10
        overall_steps = self.config["epochs"]*self.config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.Energy_schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, lr_min, overall_steps, warmup_steps)
        #optimizer = optax.adam(self.schedule)
        optimizer = optax.chain( optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.Energy_schedule(epoch)))
        return optimizer

    def compute_f_grad_and_f(self, network_params, X, pos):
        Y_func = lambda x_pos: self.parameterize_function(network_params, X, x_pos)
        vmap_Y = jax.vmap(Y_func, in_axes=(0,))( pos)

        grad_Y = lambda x_pos: jax.grad(Y_func)(x_pos)[0]
        grad_grad_Y = lambda x_pos: jax.grad(grad_Y)(x_pos)

        vmap_Y = jax.vmap(Y_func, in_axes=(0,))( pos)
        vmap_Y_grad = jax.vmap(grad_Y, in_axes=(0,))(pos)
        vmap_Y_grad_grad = jax.vmap(grad_grad_Y, in_axes=(0,))(pos)

        Y = jnp.concatenate([vmap_Y[...,None], vmap_Y_grad[...,None], vmap_Y_grad_grad, pos], axis = -1)
        return Y

    def vmap_calc_energy(self, diff_samples, energy_params, key):
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, diff_samples.shape[0])
        energy_value, batched_key = jax.vmap(self.calc_energy, in_axes=(0, None, 0))(diff_samples, energy_params, batched_key)
        return jnp.mean(energy_value), key

    def calc_energy(self, diff_samples, energy_params, key):
        y, key = self.scale_samples(diff_samples, energy_params, key)
        return self.energy_function(y), key
    
    @partial(jax.jit, static_argnums=(0,))
    def scale_samples(self, X, network_params, key):
        key, subkey = jax.random.split(key)
        pos = self.get_pos(subkey)

        Y = self.compute_f_grad_and_f(network_params, X, pos)
        return Y, key
    
    def get_pos(self, subkey):
        pos = jax.random.uniform(subkey, (self.pos_batch_size, self.pos_dim), minval = self.min_val, maxval = self.max_val)
        return pos
    
    def get_deterministic_pos(self):
        return jnp.linspace(self.min_val, self.max_val, self.pos_batch_size)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, Y, pen = 0.1):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """
        f = Y[...,0]
        f_grad = Y[..., 1]
        f_grad_grad = Y[..., 2]
        # max_f = jnp.max(jnp.abs(f))
        # where_max = jnp.where(jnp.abs(f) == max_f*jnp.ones_like(f), f, jnp.ones_like(f))
        # area_under_curve = jnp.sum(jnp.abs(f))*1/Y.shape[0]
        # target_area = 1.0
        constraint = (jnp.sqrt(f_grad**2 + f**2) - 1)**2
        penality = pen*jnp.mean(constraint)
        loss = jnp.mean((f + f_grad_grad)**2 ) + penality
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

        fig = plt.figure()
        plt.plot(Ys[...,-1], Ys[...,1], "x")
        wandb.log({"PINN/sampled_Y_grad": wandb.Image(fig)})
        plt.close()

        fig = plt.figure()
        plt.plot(Ys[...,-1], Ys[...,2], "x")
        wandb.log({"PINN/sampled_Y_grad_grad": wandb.Image(fig)})
        plt.close()

        fig = plt.figure()
        plt.plot(Ys[...,-1], Ys[...,0] - Ys[...,2], "x")
        wandb.log({"PINN/sampled_diffs": wandb.Image(fig)})
        plt.close()


    def plot_many_samples(self, diff_samples, energy_params):
        pos = self.get_deterministic_pos()[..., None]

        pos_dependent_func = lambda xx: self.parameterize_function(energy_params, diff_samples, xx)

        vmap_y = lambda xx: jax.vmap(pos_dependent_func, in_axes=(0,))(xx) 

        Ys = vmap_y(pos)
    
        return Ys
    

def learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, lr_min = 1e-4, overall_steps = 1000, warmup_steps = 100):
    cosine_decay = lambda step: optax.cosine_decay_schedule(init_value=(l_max - lr_min), decay_steps=overall_steps - warmup_steps)(step) + lr_min

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))
    

if(__name__ == "__main__"):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{5}"


    config = {"epochs": 10000, "steps_per_epoch": 1, "lr_factor": 1, "project_name": "vanilla_PINN", "lr": 3*10**-3}
    model = WavePINN_latent_Class(config)
    model.train()

