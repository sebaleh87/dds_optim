from .FeedForward import FeedForwardNetwork, FourierNetwork, EncodingNetwork
from .LSTM import LSTMNetwork
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax

NetworkRegistry = {"FeedForward": FeedForwardNetwork, "FourierNetwork": FourierNetwork, "LSTMNetwork": LSTMNetwork}


def get_network(network_config, SDE_Loss_Config):
    return NetworkRegistry[network_config["name"]](n_layers=network_config["n_layers"], hidden_dim=network_config["n_hidden"])

class BaseModel(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.encoding_network = EncodingNetwork(feature_dim=self.network_config["feature_dim"], max_time = self.SDE_Loss_Config["n_integration_steps"])
        self.backbone = get_network(self.network_config, self.SDE_Loss_Config)
        self.time_backbone = get_network(self.network_config, self.SDE_Loss_Config)
        self.use_normal = False
        
    @nn.compact
    def __call__(self, in_dict):
        ### TODO compute embedding here?
        in_dict["grads"] = jax.lax.stop_gradient(in_dict["grads"])
        encoding = self.encoding_network(in_dict)
        in_dict["encoding"] = encoding

        out_dict = self.backbone(in_dict)
        embedding = out_dict["embedding"]

        x_dim = in_dict["x"].shape[-1]
        if(self.SDE_mode == "DiscreteTime_SDE"):
            mean_x = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            log_var_x = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                  bias_init=nn.initializers.zeros)(embedding)

            out_dict["mean_x"] = mean_x
            out_dict["log_var"] = log_var_x
            return out_dict
        elif(self.use_interpol_gradient and self.use_normal):
            grads = in_dict["grads"]
            t = in_dict["t"]
            time_in_dict = {"t": t, "grads": t, "x": t} 
            time_out_dict = self.time_backbone(time_in_dict)
            time_embedding = time_out_dict["embedding"]
            grad_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(time_embedding)
            
            correction_drift = nn.Dense(x_dim, kernel_init=nn.initializers.xavier_normal(),
                                                bias_init=nn.initializers.zeros)(embedding)
            
            
            grad_score = grad_drift * jnp.clip(grads, -10**2, 10**2) #* nn.softplus(interpolated_grad) 
            correction_grad_score = correction_drift + grad_score
            score = jnp.clip(correction_grad_score, -10**4, 10**4 )

            out_dict["score"] = score
            return out_dict
        elif(self.use_interpol_gradient and not self.use_normal):
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