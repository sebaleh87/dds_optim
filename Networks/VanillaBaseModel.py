from .EncodingNetworks import EncodingNetwork
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax
from .model_registry import get_network


class VanillaBaseModelClass(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.encoding_network = EncodingNetwork(feature_dim=self.network_config["feature_dim"], hidden_dim=self.network_config["n_hidden"], max_time = self.SDE_Loss_Config["n_integration_steps"])
        self.backbone = get_network(self.network_config, self.SDE_Loss_Config)
        self.time_backbone = get_network(self.network_config, self.SDE_Loss_Config)
        self.use_normal = self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        self.model_mode = self.network_config["model_mode"] # is either normal or latent

        self.x_dim = self.network_config["x_dim"]
        if(self.model_mode == "latent"):
            self.latent_dim = self.network_config["latent_dim"]
            self.encode_model = get_network(self.network_config, self.SDE_Loss_Config)
            self.decode_model = get_network(self.network_config, self.SDE_Loss_Config)

        
    @nn.compact
    def __call__(self, in_dict, train = False, forw_mode = "diffusion"): # forw_mode is either diffusion, encode, or decode
        if(self.model_mode == "normal"):
            return self.normal_forward_pass(in_dict, train = train)
        elif(self.model_mode == "latent"):
            if(forw_mode == "diffusion"):
                return self.normal_forward_pass(in_dict, train = train)
            elif(forw_mode == "encode"):
                return self.encode(in_dict)
            elif(forw_mode == "decode"):
                return self.decode(in_dict)
            elif(forw_mode == "init"):
                return self.normal_forward_pass(in_dict, train = train), self.encode(in_dict), self.decode(in_dict)
        else:
            raise ValueError(f"Unknown model_mode: {self.model_mode}")

    def encode(self, in_dict):
        in_dict["encoding"] = in_dict["x"]
        out_dict = self.encode_model(in_dict)
        embedding = out_dict["embedding"]
        mean_z = nn.Dense(self.latent_dim, kernel_init=nn.initializers.xavier_normal(),
                                            bias_init=nn.initializers.zeros)(embedding)
        log_var_z = nn.Dense(self.latent_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)

        out_dict["mean_z"] = mean_z
        out_dict["log_var_z"] = log_var_z
        return out_dict

    def decode(self, in_dict):
        in_dict["encoding"] = in_dict["z"]
        out_dict = self.encode_model(in_dict)
        embedding = out_dict["embedding"]
        mean_x = nn.Dense(self.x_dim, kernel_init=nn.initializers.xavier_normal(),
                                            bias_init=nn.initializers.zeros)(embedding)
        log_var_x = nn.Dense(self.x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)

        out_dict["mean_x"] = mean_x
        out_dict["log_var_x"] = log_var_x
        return out_dict

    def normal_forward_pass(self, in_dict, train = False):

        copy_grads = in_dict["grads"]
        if(self.use_normal or self.SDE_mode == "Bridge_SDE"):
            pass
            #in_dict["grads"] = jnp.zeros_like(in_dict["grads"])
            #in_dict["Energy_value"] = jnp.zeros_like(in_dict["Energy_value"])
        else:
            ### parametrization as in Learning to learn by gradient descent by gradient descent
            grad = in_dict["grads"]
            Energy = jnp.zeros_like(in_dict["Energy_value"])
            eps = 10**-10
            scaled_energy = Energy
            p = 10
            scaled_energy_2 = jnp.where(jnp.abs(scaled_energy) >= jnp.exp(-p), jnp.log(jnp.abs(scaled_energy) + eps)/p, jnp.exp(p)*scaled_energy)
            scaled_energy_1 = jnp.where(jnp.abs(scaled_energy) >= jnp.exp(-p),  jnp.sign(scaled_energy), -1)
            grad_2 = jnp.where(jnp.abs(grad) >= jnp.exp(-p), jnp.log(jnp.abs(grad) + eps)/p, jnp.exp(p)*grad)
            grad_1 = jnp.where(jnp.abs(grad) >= jnp.exp(-p),  jnp.sign(grad), -1)
            in_dict["grads"] = jnp.concatenate([grad_1, grad_2], axis = -1) ### TODO pay attention once there was a stop gradient, not it on x is before grads are computen
            in_dict["Energy_value"] = jnp.concatenate([scaled_energy_1, scaled_energy_2], axis = -1)



        encoding = self.encoding_network(in_dict, train = train, use_normal = self.use_normal)
        in_dict["encoding"] = encoding

        out_dict = self.backbone(in_dict)
        embedding = out_dict["embedding"]

        x_dim = in_dict["x"].shape[-1]
        if(self.SDE_mode == "Bridge_SDE" and self.use_normal):
            # follows SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS (32)
            grad = copy_grads
            score = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)

            out_dict["score"] = score  + grad /2     
            return out_dict
        elif(self.SDE_mode == "Bridge_SDE" and not self.use_normal):
            grads = copy_grads

            grad_drift = nn.Dense(x_dim, kernel_init=nn.initializers.zeros,
                                                bias_init=nn.initializers.zeros)(embedding)
            
            correction_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            
            grad_score = grad_drift * jnp.clip(grads, -10**2, 10**2) #* nn.softplus(interpolated_grad) 
            correction_grad_score = correction_drift + grad_score
            score = jnp.clip(correction_grad_score, -10**4, 10**4 )

            out_dict["score"] = score
            return out_dict
        elif(self.SDE_mode == "DiscreteTime_SDE"):
            mean_x = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            log_var_x = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                  bias_init=nn.initializers.zeros)(embedding)

            out_dict["mean_x"] = mean_x
            out_dict["log_var"] = log_var_x
            return out_dict
        elif(self.use_interpol_gradient and self.use_normal):
            #this follows http://arxiv.org/abs/2302.13834 equation (88)
            
            grads = copy_grads

            grad_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            
            correction_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            
            grad_score = grad_drift * jnp.clip(grads, -10**2, 10**2) #* nn.softplus(interpolated_grad) 
            correction_grad_score = correction_drift + grad_score
            score = jnp.clip(correction_grad_score, -10**4, 10**4 )

            out_dict["score"] = score
            return out_dict
        elif(self.use_interpol_gradient and not self.use_normal):
            #print(jnp.mean(grad), jnp.mean(in_dict["grads"]))
            if(self.network_config["name"] == "ADAMNetwork"):
                score = embedding
                out_dict["score"] = score            
                return out_dict
            else:
                correction_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                    bias_init=nn.initializers.zeros)(embedding)
                
                correction_grad_score = correction_drift 
                score = jnp.clip(correction_grad_score, -10**4, 10**4 )
                out_dict["score"] = score

                return out_dict
        else:
            score = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            out_dict["score"] = score            
            return out_dict