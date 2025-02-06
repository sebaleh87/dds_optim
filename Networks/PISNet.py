from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax

class PISNetBaseClass(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    num_layers: int = 2
    num_hid: int = 90
    out_clip: float = 10**4


    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.beta_schedule = self.SDE_Loss_Config["SDE_Type_Config"]["beta_schedule"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.use_normal = self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        self.model_mode = self.network_config["model_mode"] # is either normal or latent
        self.n_integration_steps = self.SDE_Loss_Config["n_integration_steps"]

        #self.num_hid = self.network_config["n_hidden"]
        self.x_dim = self.network_config["x_dim"]
        self.dim = self.x_dim

        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.state_time_coder = TimeEncoder(self.num_hid)
        self.state_time_net = StateTimeEncoder(num_layers=self.num_layers, out_dim = self.dim,  num_hid=self.num_hid,
                                               name='state_time_net', parent=self)
        
        self.out_layer = nn.Dense(self.x_dim, kernel_init=nn.initializers.zeros,
                                                bias_init=nn.initializers.zeros)
        self.beta_layer = nn.Dense(self.x_dim, kernel_init=nn.initializers.zeros,
                                                bias_init=nn.initializers.zeros)

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, in_dict, train = False, forw_mode = "diffusion"):
        out_dict = {}
        input_array = in_dict["x"]
        time_array = in_dict["t"]*self.n_integration_steps
        grad = in_dict["grads"]

        t_embed = self.state_time_coder(time_array)

        extended_input = jnp.concatenate((input_array, t_embed), axis=-1)
        extended_output = self.state_time_net(extended_input)
        
        out_state = self.out_layer(extended_output)

        out_state_p_grad =  jnp.clip(out_state, -self.out_clip, self.out_clip)
        out_score =  out_state_p_grad + grad/2
        out_dict["score"] = out_score   

        if(self.beta_schedule == "neural"):
            log_beta_x_t = self.beta_layer(extended_output)
            out_dict["log_beta_x_t"] = log_beta_x_t
        return out_dict


class TimeEncoder(nn.Module):
    num_hid: int = 90

    def setup(self):
        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.mlp = [
            nn.Dense(self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
        ]

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, time_array_emb):
        time_array_emb = self.get_fourier_features(time_array_emb)
        for layer in self.mlp:
            time_array_emb = layer(time_array_emb)
        return time_array_emb


class StateTimeEncoder(nn.Module):
    num_layers: int = 2
    out_dim: int = 2
    num_hid: int = 64
    zero_init: bool = False

    def setup(self):
        self.mlp = [
            nn.Dense(self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
            nn.gelu,
        ]
        self.out_layer = nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros,
                                                bias_init=nn.initializers.zeros)

    def __call__(self, extended_input):
        for layer in self.mlp:
            extended_input = layer(extended_input)

        return extended_input
