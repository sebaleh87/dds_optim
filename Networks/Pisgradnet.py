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

    bias_init: float = 0.

    def setup(self):
        self.SDE_mode = self.SDE_Loss_Config["SDE_Type_Config"]["name"]
        self.bridge_type = self.SDE_Loss_Config["SDE_Type_Config"].get("bridge_type", "CMCD")
        self.beta_schedule = self.SDE_Loss_Config["SDE_Type_Config"]["beta_schedule"]
        self.use_interpol_gradient = self.SDE_Loss_Config["SDE_Type_Config"]["use_interpol_gradient"]
        self.use_normal = self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        self.model_mode = self.network_config["model_mode"] # is either normal or latent
        self.n_integration_steps = self.SDE_Loss_Config["n_integration_steps"]
        self.network_init = self.network_config["network_init"]
        self.langevin_precon = self.network_config["langevin_precon"]

        self.weight_init = 1e-8

        #self.num_hid = self.network_config["n_hidden"]
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
        
        self.time_coder_grad_zero_init = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
            [nn.gelu, nn.Dense(self.num_hid, kernel_init=nn.initializers.zeros,
                                                          bias_init=nn.initializers.zeros)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.zeros,
                                                          bias_init=nn.initializers.zeros)])

        self.state_time_net = nn.Sequential([nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                         bias_init=nn.initializers.zeros_init())])
        
        if(self.beta_schedule == "neural"):
            self.reverse_time_grad = self.time_coder_grad_zero_init
        else:
            self.reverse_time_grad = self.time_coder_grad


        if(self.bridge_type == "DBS"):
            self.forw_state_time_net = nn.Sequential([nn.Sequential(
                                    [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                         bias_init=nn.initializers.zeros_init())])
            if(self.beta_schedule == "neural"):
                self.forward_time_grad  = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
                                            [nn.gelu, nn.Dense(self.num_hid, kernel_init=nn.initializers.zeros,
                                                          bias_init=nn.initializers.zeros)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.zeros,
                                                          bias_init=nn.initializers.zeros)])


            else:
                self.forward_time_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
                                                        [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                          bias_init=nn.initializers.constant(self.bias_init))])


    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)
    
    def parameterize_score(self, out_dict, out_state, time_array_emb, grad, lgv_term, time_grad_func):
        if(self.langevin_precon):
            if(self.beta_schedule == "neural"):
                t_net2 = time_grad_func(time_array_emb)
                out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
                log_beta_x_t = t_net2
                out_dict["log_beta_x_t"] = log_beta_x_t
                correction_grad_score = out_state 
                score = jnp.clip(correction_grad_score, -10**4, 10**4 )
            else:
                
                t_net2 = time_grad_func(time_array_emb)
                out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
                lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
                score = out_state + t_net2 * lgv_term

            if(self.bridge_type == "CMCD"):
                overall_score = score  + grad /2    
            else:
                overall_score = score  
        else:
            if(self.beta_schedule == "neural"):
                t_net2 = time_grad_func(time_array_emb)
                out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
                log_beta_x_t = t_net2
                out_dict["log_beta_x_t"] = log_beta_x_t
                correction_grad_score = out_state 
                score = jnp.clip(correction_grad_score, -10**4, 10**4 )
            else:
                out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
                score = out_state

        return overall_score, out_dict

    def __call__(self, in_dict, train = False, forw_mode = "diffusion"):
        out_dict = {}
        input_array = in_dict["x"]
        time_array = in_dict["t"]*self.n_integration_steps
        lgv_term = jax.lax.stop_gradient(in_dict["grads_T1"])
        grad = in_dict["grads"]

        time_array_emb = self.get_fourier_features(time_array)

        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)

        overall_score, out_dict = self.parameterize_score( out_dict, out_state, time_array_emb, grad, lgv_term, self.reverse_time_grad)
        out_dict["score"] = overall_score

        if(self.bridge_type == "DBS"):
            forw_out_state = self.forw_state_time_net(extended_input)
            forward_score, out_dict = self.parameterize_score(out_dict, forw_out_state, time_array_emb, grad, lgv_term, self.forward_time_grad)
            out_dict["forward_score"] = forward_score

        return out_dict