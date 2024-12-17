from .EncodingNetworks import TimeEncodingNetwork
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax
from .EGNN import EGNNLayer, EGNNNetwork


class EGNNBaseClass(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.encoding_network = TimeEncodingNetwork(feature_dim=self.network_config["feature_dim"], hidden_dim=self.network_config["n_hidden"], max_time = self.SDE_Loss_Config["n_integration_steps"])
        self.backbone = EGNNNetwork(n_layers = self.network_config["n_layers"], hidden_dim = self.network_config["n_hidden"], 
                                    feature_dim = self.network_config["feature_dim"], n_particles = self.network_config["n_particles"],
                                    out_dim = self.network_config["out_dim"])
        self.use_normal = self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        self.n_particles = self.network_config["n_particles"]
        self.particle_dim = self.network_config["out_dim"]
        
    @nn.compact
    def __call__(self, in_dict, train = False):
        ### TODO rewrite code so that __call__ is allways applied with vmap
        if(self.use_normal):
            copy_grads = jax.lax.stop_gradient(in_dict["grads"])
            in_dict["grads"] = jnp.zeros_like(in_dict["grads"])
            in_dict["Energy_value"] = jax.lax.stop_gradient(jnp.zeros_like(in_dict["Energy_value"]))
        else:
            ### TODO compute grad norm to keep this invariant
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

        #print([(key, in_dict[key].shape) for key in in_dict.keys()])
        out_dict = jax.vmap(self.backbone, in_axes = (0,))(in_dict)
        x_score = out_dict["x"]
        h_score = out_dict["x_hidden"]
        
        grads = copy_grads

        if(self.use_normal):
            clip_value = 20
            clip_overall_score = 10**4
            grads_norm = jnp.linalg.norm(grads, axis = -1, keepdims = True)
            clipped_grads = jnp.where(grads_norm > clip_value, clip_value*grads/grads_norm, grads)
            
            grad_score = h_score * clipped_grads
            correction_grad_score = x_score+ grad_score
            correction_grad_score_norm = jnp.linalg.norm(correction_grad_score, axis = -1, keepdims = True)
            clipped_score = jnp.where(correction_grad_score_norm > clip_overall_score, clip_overall_score*correction_grad_score/correction_grad_score_norm, correction_grad_score)

            resh_score = clipped_score.reshape(-1, self.n_particles, self.particle_dim)
            resh_COM_score = resh_score - jnp.mean(resh_score, axis = 1, keepdims=   True)
            COM_score = resh_COM_score.reshape(-1, self.particle_dim*self.n_particles)
            out_dict["score"] = COM_score
            return out_dict
        else:
            out_dict["score"] = x_score
            return out_dict


