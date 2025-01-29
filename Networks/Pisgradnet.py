from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax

class PisgradnetBaseClass(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.use_normal = self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        self.model_mode = self.network_config["model_mode"] # is either normal or latent
        self.n_integration_steps = self.SDE_Loss_Config["n_integration_steps"]

        self.x_dim = self.network_config["x_dim"]
        self.dim = self.x_dim

        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
        ])

        self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
            [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                          bias_init=nn.initializers.constant(self.bias_init))])

        self.state_time_net = nn.Sequential([nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
                                                         bias_init=nn.initializers.zeros_init())])

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
        lgv_term = jax.lax.stop_gradient(in_dict["grads"])
        grad = in_dict["grads"]

        time_array_emb = self.get_fourier_features(time_array)
        # if len(input_array.shape) == 1:
        #     time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        t_net2 = self.time_coder_grad(time_array_emb)

        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        out_state_p_grad = out_state + t_net2 * lgv_term
        out_score =  out_state_p_grad + grad/2
        out_dict["score"] = out_score   
        return out_dict