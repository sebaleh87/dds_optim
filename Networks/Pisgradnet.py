from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax

class PisgradnetBaseClass(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

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
        self.beta_schedule_neural_mode = self.network_config.get("beta_schedule_neural_mode", "time_dependent")

        self.weight_init = self.network_config.get("weight_init", 1e-8)
        self.num_hid = self.network_config.get("n_hidden", 64)
        self.num_layers = self.network_config.get("n_layers", 2)

        self.compute_value_func = self.network_config.get("compute_value_func", False)
        if(self.SDE_Loss_Config["name"] == "PPO_Loss"):
            self.compute_value_func = True

        #self.num_hid = self.network_config["n_hidden"]
        self.x_dim = self.network_config["x_dim"]
        self.dim = self.x_dim

        self.diff_function_dict = self.create_diffusion_network(diff_direction = "reverse")
        if(self.bridge_type == "DBS"):
            self.diff_function_dict_forward = self.create_diffusion_network(diff_direction = "forward")
                
    def create_diffusion_network(self , diff_direction = "reverse"):
        ### for CMCD reverse and forward directions are coupled, so also diff_direction reverse will be used
        ### for DBS reverse and forward directions are decoupled, so diff_direction forward will be used
        time_step_phase = self.param(f'{diff_direction}_timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        time_step_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        NN_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
            [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                          bias_init=nn.initializers.constant(self.bias_init))])
        
        NN_score_time_embedding_net = nn.Sequential([
                                        nn.Dense(self.num_hid),
                                        nn.gelu,
                                        nn.Dense(self.num_hid),
                                    ])
        
        NN_score = nn.Sequential([nn.Sequential(
                                    [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                         bias_init=nn.initializers.zeros_init())])


        diff_function_out_dict = {"time_step_phase": time_step_phase, "time_step_coeff": time_step_coeff,
                                  "NN_grad": NN_grad, "NN_score": NN_score, "NN_score_time_embedding_net": NN_score_time_embedding_net}
        
        if(self.beta_schedule == "neural"):
            beta_schedule_network = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
                                                [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                          bias_init=nn.initializers.constant(self.bias_init))])
            diff_function_out_dict["beta_schedule_network"] = beta_schedule_network

        if(self.compute_value_func):
            value_function = nn.Sequential([nn.Sequential(
                                    [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                nn.Dense(1, kernel_init=nn.initializers.constant(self.weight_init),
                                                         bias_init=nn.initializers.zeros_init())])
            diff_function_out_dict["value_function"] = value_function


        return diff_function_out_dict
    
    def get_fourier_features(self, diff_function_out_dict, timesteps):
        timestep_coeff = diff_function_out_dict["time_step_coeff"]
        timestep_phase = diff_function_out_dict["time_step_phase"]
        sin_embed_cond = jnp.sin(
            (timestep_coeff * timesteps) + timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (timestep_coeff * timesteps) + timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)
    
    def compute_score_inputs(self, in_dict, diff_function_out_dict):
        input_array = in_dict["x"]
        time_array = in_dict["t"]*self.n_integration_steps
        stopped_grad = jax.lax.stop_gradient(in_dict["grads_T1"])
        grad = in_dict["grads"]
        
        time_array_emb = self.get_fourier_features(diff_function_out_dict, time_array)

        NN_score_time_embedding_net = diff_function_out_dict["NN_score_time_embedding_net"]
        time_embedding = NN_score_time_embedding_net(time_array_emb)

        extended_input = jnp.concatenate((input_array, time_embedding), axis=-1)
        NN_score = diff_function_out_dict["NN_score"]
        out_state = NN_score(extended_input)
        beta_net_input = self.create_t2_net_input(input_array, time_array_emb)    

        embedding_dict = {"out_state": out_state, "stopped_grad": stopped_grad, "time_array_emb": time_array_emb,
                          "NN_grad_input": time_array_emb, "beta_net_input": beta_net_input, "grad": grad}

        if(self.compute_value_func):
            value_function = diff_function_out_dict["value_function"]
            value_function_value = value_function(extended_input)
            embedding_dict["value_function_value"] = value_function_value

        return embedding_dict
    
    def parameterize_score(self, out_dict, diff_function_out_dict, embedding_dict):
        out_state = embedding_dict["out_state"]
        NN_grad_input = embedding_dict["NN_grad_input"]     
        beta_net_input = embedding_dict["beta_net_input"] 
        grad = embedding_dict["grad"]
        lgv_term = embedding_dict["stopped_grad"]

        NN_grad = diff_function_out_dict["NN_grad"]
        if(self.langevin_precon):
            t_net2 = NN_grad(NN_grad_input)
            out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
            lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
            score = out_state + t_net2 * lgv_term

            if(self.bridge_type == "CMCD"):
                score = score  + grad /2    
            else:
                pass
        else:
            out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
            score = out_state

        if(self.beta_schedule == "neural"):
            beta_schedule_network = diff_function_out_dict["beta_schedule_network"]
            log_beta_x_t = beta_schedule_network(beta_net_input)
            out_dict["log_beta_x_t"] = log_beta_x_t

        return score, out_dict
    
    def create_t2_net_input(self, input_array, time_array_emb):
        if(self.beta_schedule_neural_mode == "time_dependent"):
            t2_net_input = time_array_emb
        elif self.beta_schedule == "neural":
            t2_net_input = jnp.concatenate([input_array, time_array_emb], axis=-1)
        else:
            raise ValueError("Config of beta_schedule and langevin_precon_mode are incompatible")  # Raise an error for unsupported configurations

        return t2_net_input


    def __call__(self, in_dict, train = False, forw_mode = "diffusion"):
        out_dict = {}
        embedding_dict = self.compute_score_inputs(in_dict, self.diff_function_dict)
        overall_score, out_dict = self.parameterize_score( out_dict, self.diff_function_dict, embedding_dict)
        out_dict["score"] = overall_score

        out_dict["value_function_value"] = embedding_dict["value_function_value"] if self.compute_value_func else None

        if(self.bridge_type == "DBS"):
            embedding_dict = self.compute_score_inputs(in_dict, self.diff_function_dict_forward)
            forward_score, out_dict = self.parameterize_score( out_dict, self.diff_function_dict_forward, embedding_dict)
            out_dict["forward_score"] = forward_score

        return out_dict