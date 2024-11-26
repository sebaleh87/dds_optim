from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import flax.linen as nn

class FeedForwardNetwork(nn.Module):
    n_layers: int = 3
    hidden_dim: int = 32
    n_out: int = 2

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x):
        for _ in range(self.n_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
        
        x = nn.Dense(self.n_out, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)(x)
        
        x = nn.sigmoid(x)
        return x

class DoubleMoon(EnergyModelClass):
    def __init__(self, config ):
        super().__init__(config)
        self.data = self.create_double_moon_dataset(1000)
        self.z_dim = 2
        self.model = FeedForwardNetwork( n_layers = 3, hidden_dim = 32, n_out = 2)
        self.x_dim = None ### TODO compute x_dim based on lora
        self.lora_dim = 6
        self.data_batch_size = 1000

    def create_double_moon_dataset(self, n_samples, noise=0.1):
        n_samples_per_moon = n_samples // 2
        outer_circ_x = jnp.cos(jnp.linspace(0, jnp.pi, n_samples_per_moon))
        outer_circ_y = jnp.sin(jnp.linspace(0, jnp.pi, n_samples_per_moon))

        inner_circ_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, n_samples_per_moon))
        inner_circ_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, n_samples_per_moon)) - .5

        outer_circ = jnp.vstack([outer_circ_x, outer_circ_y]).T
        inner_circ = jnp.vstack([inner_circ_x, inner_circ_y]).T

        data = jnp.concatenate([outer_circ, inner_circ], axis=0)
        data += noise * jax.random.normal(jax.random.PRNGKey(0), data.shape)

        labels = jnp.concatenate([jnp.zeros((n_samples_per_moon,1)), jnp.ones((n_samples_per_moon, 1))], axis = 0)

        return data, labels
    
    def get_random_data_sample(self, batch_size, key):
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(subkey, self.data.shape[0], (batch_size,), replace=False)
        selected_data = self.data[indices]
        z = selected_data
        Y_target = selected_data
        return {"z": z, "Y_target": Y_target}, key

        pass

    def calculate_model_output(self, pred_params, z):
        ### TODO project lora params to param dimension and add params
        return self.model.apply(pred_params, z)

    def _init_params(self):
        self.init_params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.z_dim)))

        self.num_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, self.init_params["params"])))
        self.x_dim = self.num_params
        # params_dict_shape = jax.tree_util.tree_map(lambda x: x.shape, self.init_params["params"])
        # self.lora_param_A = 
        # self.lora_param_B = 

    def map_to_params(self, flat_params):
        def unflatten(flat_params, tree):
            leaves, treedef = jax.tree_util.tree_flatten(tree)
            sizes = jnp.array([leaf.size for leaf in leaves])
            flat_leaves = jnp.split(flat_params, jnp.cumsum(sizes)[:-1])
            return jax.tree_util.tree_unflatten(treedef, [jnp.reshape(flat_leaf, leaf.shape) for flat_leaf, leaf in zip(flat_leaves, leaves)])

        return unflatten(flat_params, self.init_params["params"])

    def calc_energy(self, diff_samples, energy_params, key):
        Y_pred, Y_target, key = self.scale_samples(diff_samples, energy_params, key)

        return self.energy_function(Y_pred, Y_target), key
    
    def scale_samples(self, diff_samples, energy_params, key):
        data_dict, key = self.get_random_data_sample(self.data_batch_size, key)
        z = data_dict["z"]
        Y_target = data_dict["Y_target"]
        Y_pred = self.model.apply(diff_samples, z)
        return Y_pred, Y_target, key

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, Y_pred, Y_target):

        return jnp.mean((Y_pred - Y_target)**2)
    

