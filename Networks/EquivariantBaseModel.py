from .EncodingNetworks import TimeEncodingNetwork
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax
from .EGNN import EGNNLayer


class EGNNBaseClass(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.encoding_network = TimeEncodingNetwork(feature_dim=self.network_config["feature_dim"], hidden_dim=self.network_config["n_hidden"], max_time = self.SDE_Loss_Config["n_integration_steps"])
        self.backbone = EGNNLayer(self.network_config["n_layers"], self.network_config["n_hidden"], self.network_config["feature_dim"], self.network_config["n_particles"])
        self.use_normal = True#self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        
    @nn.compact
    def __call__(self, in_dict, train = False):
        ### TODO compute embedding here?
        if(self.use_normal):
            copy_grads = jax.lax.stop_gradient(in_dict["grads"])
            in_dict["grads"] = jnp.zeros_like(in_dict["grads"])
            in_dict["Energy_value"] = jax.lax.stop_gradient(jnp.zeros_like(in_dict["Energy_value"]))
        else:
            grad = in_dict["grads"]
            Energy = jax.lax.stop_gradient(jnp.zeros_like(in_dict["Energy_value"]))
            eps = 10**-10
            scaled_energy = Energy
            p = 10
            scaled_energy_2 = jnp.where(jnp.abs(scaled_energy) >= jnp.exp(-p), jnp.log(jnp.abs(scaled_energy) + eps)/p, jnp.exp(p)*scaled_energy)
            scaled_energy_1 = jnp.where(jnp.abs(scaled_energy) >= jnp.exp(-p),  jnp.sign(scaled_energy), -1)
            grad_2 = jnp.where(jnp.abs(grad) >= jnp.exp(-p), jnp.log(jnp.abs(grad) + eps)/p, jnp.exp(p)*grad)
            grad_1 = jnp.where(jnp.abs(grad) >= jnp.exp(-p),  jnp.sign(grad), -1)
            in_dict["grads"] = jax.lax.stop_gradient(jnp.concatenate([grad_1, grad_2], axis = -1))
            in_dict["Energy_value"] = jax.lax.stop_gradient(jnp.concatenate([scaled_energy_1, scaled_energy_2], axis = -1))


        ### TODO only encode time here
        encoding = self.encoding_network(in_dict, train = train)
        in_dict["h"] = encoding

        out_dict = self.backbone(in_dict)
        embedding = out_dict["x"]
        time_embedding = out_dict["h"]

        x_dim = in_dict["x"].shape[-1]
        
        grads = copy_grads

        grad_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                            bias_init=nn.initializers.zeros)(time_embedding)
        
        correction_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                            bias_init=nn.initializers.zeros)(embedding)
        
        grad_score = grad_drift * jnp.clip(grads, -10**2, 10**2) #* nn.softplus(interpolated_grad) 
        correction_grad_score = correction_drift + grad_score
        score = jnp.clip(correction_grad_score, -10**4, 10**4 )

        out_dict["score"] = score
        return out_dict


