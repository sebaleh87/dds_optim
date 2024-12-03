from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import flax.linen as nn
import wandb

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
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        x = nn.Dense(self.n_out, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros)(x)
        
        x = nn.log_softmax(x)
        return x

class DoubleMoonClass(EnergyModelClass):
    def __init__(self, config ):
        self.overall_data_size = 1000
        self.data_z, self.labels = self.create_double_moon_dataset(self.overall_data_size)
        self.z_dim = 2
        self.model = FeedForwardNetwork( n_layers = 4, hidden_dim = 32, n_out = 2)
        self.vmap_apply = jax.vmap(self.model.apply, in_axes=(None, 0))
        self._init_params()
        self.lora_dim = 6
        self.data_batch_size = 200
        self.diff_param_scale = 0.1
        config["dim_x"] = self.num_params
        print("Attention L2 regularization not used (prior regularization)")
        super().__init__(config)

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
        indices = jax.random.choice(subkey, self.data_z.shape[0], (batch_size,), replace=False)
        selected_data = self.data_z[indices]
        selected_labels = self.labels[indices]
        z = selected_data
        Y_target = selected_labels
        return {"z": z, "Y_target": Y_target}, key


    def calculate_model_output(self, pred_params, z):
        ### TODO project lora params to param dimension and add params
        return self.model.apply(pred_params, z)

    def _init_params(self):
        self.init_params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.z_dim)))

        self.num_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, self.init_params["params"])))
        self.x_dim = self.num_params

        self.leaves, self.treedef = jax.tree_util.tree_flatten(self.init_params["params"])
        self.sizes = jnp.array([leaf.size for leaf in self.leaves])
        self.cumsum = jax.lax.cumsum(self.sizes)[:-1].tolist()
        self.flat_init_params = jnp.concatenate([leaf.flatten() for leaf in self.leaves])
        # params_dict_shape = jax.tree_util.tree_map(lambda x: x.shape, self.init_params["params"])
        # self.lora_param_A = 
        # self.lora_param_B = 

    def map_to_params(self, flat_params):
        def unflatten(flat_params):
            flat_leaves = jnp.array_split(flat_params, self.cumsum )
            return jax.tree_util.tree_unflatten(self.treedef, [jnp.reshape(flat_leaf, leaf.shape) for flat_leaf, leaf in zip(flat_leaves, self.leaves)])

        return {"params": unflatten(flat_params)}

    def calc_energy(self, diff_samples, energy_params, key):
        Y_pred_Y_target, key = self.scale_samples(diff_samples, energy_params, key)

        data_energy = self.energy_function(Y_pred_Y_target)

        overall_params = diff_samples # self.flat_init_params  + self.diff_param_scale *diff_samples
        param_energy = -jnp.sum(jax.scipy.stats.norm.logpdf(overall_params, loc=0, scale=1) ) 
        overall_energy = data_energy + param_energy
        return overall_energy, key
    
    def scale_samples(self, diff_samples, energy_params, key):
        data_dict, key = self.get_random_data_sample(self.data_batch_size, key)
        z = data_dict["z"]
        Y_target = data_dict["Y_target"]

        diff_params = self.map_to_params(diff_samples)
        combined_params = diff_params
        #combined_params = jax.tree_util.tree_map(lambda x, y: x + self.diff_param_scale *y, self.init_params, diff_params)
        ### TODO use xavier params as init and add diff_params to it
        Y_pred = self.vmap_apply(combined_params, z)

        #print("shapes", Y_pred.shape, Y_target.shape, z.shape, diff_samples.shape)
        ### TODO vmap this

        return jnp.concatenate([Y_pred, Y_target], axis = -1), key

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, Y_pred_Y_target):
        log_Y_pred = Y_pred_Y_target[:,0:2]
        Y_target = Y_pred_Y_target[:,-1]
        CrossEntropy = lambda log_Y_pred, y_target: -self.overall_data_size*jnp.mean(y_target*log_Y_pred[:,0] + (1-y_target)*log_Y_pred[:,1])
        return CrossEntropy(log_Y_pred, Y_target)
    
    def visualize_models(self, diff_samples):
        fig, axs = plt.subplots(diff_samples.shape[0], 2, figsize=(12, 6))
        axs = axs.flatten()
        counter = 0
        for idx, diff_sample in enumerate(diff_samples):
            diff_params = self.map_to_params(diff_sample)

            full_z_samples = self.data_z
            log_Y_pred = self.vmap_apply(diff_params, full_z_samples)
            Y_pred = jnp.exp(log_Y_pred)[:,0]
            
            #fig, axs = plt.subplots(idx + 1, 2, figsize=(12, 6))

            scatter1 = axs[counter].scatter(full_z_samples[:, 0], full_z_samples[:, 1], c=Y_pred, cmap='viridis', s = 4)
            axs[counter].set_title('Predicted Labels')
            fig.colorbar(scatter1, ax=axs[counter])
            counter += 1

            scatter2 = axs[counter].scatter(full_z_samples[:, 0], full_z_samples[:, 1], c=self.labels.flatten(), cmap='viridis', s = 4)
            axs[counter].set_title('True Labels')
            fig.colorbar(scatter2, ax=axs[counter])
            counter += 1

        wandb.log({"figs/DoubleMoon": wandb.Image(fig)})
    
        plt.close()

    def estimate_uncertainty(self, ):
        pass

