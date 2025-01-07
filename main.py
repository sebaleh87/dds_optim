from EnergyFunctions import Rastrigin
import jax
from jax import config as jax_config
import os
from Trainer.train import TrainerClass
import argparse
import numpy as np
import torch
### TODO make seperate run configs for discrete time and continuous time

parser = argparse.ArgumentParser(description="Denoising Diffusion Sampler")
parser.add_argument("--GPU", type=int, default=6, help="GPU id to use")
parser.add_argument("--SDE_Loss", type=str, default="LogVariance_Loss", choices=["Reverse_KL_Loss","LogVariance_Loss", "LogVariance_Loss_MC", 
                                                                                 "LogVariance_Loss_with_grad", "LogVariance_Loss_weighted",
                                                                                 "Bridge_rKL", "Bridge_LogVarLoss",
                                                                                "Discrete_Time_rKL_Loss_log_deriv", "Discrete_Time_rKL_Loss_reparam"], help="select loss function")
parser.add_argument("--SDE_Type", type=str, default="VP_SDE", choices=["VP_SDE", "subVP_SDE", "VE_SDE", "Bridge_SDE"], help="GPU id to use")
parser.add_argument("--Energy_Config", type=str, default="GaussianMixture", choices=["GaussianMixture", "GaussianMixtureToy", "Rastrigin", "LennardJones", "DoubleWell_iter", "DoubleWell_Richter",
                                                                                     "MexicanHat", "Pytheus", "WavePINN_latent", "WavePINN_hyperparam", "DoubleMoon"], help="EnergyClass")
parser.add_argument("--T_start", type=float, default=1., help="Starting Temperature")
parser.add_argument("--T_end", type=float, default=0., help="End Temperature")
parser.add_argument("--anneal_lam", type=float, default=10., help="Strech anneal schedule; not possible for linear schedule")
parser.add_argument("--n_integration_steps", type=int, default=100)
parser.add_argument("--SDE_weightening", type=str, default="normal", choices=["normal", "weighted"], help="SDE weightening")
parser.add_argument("--AnnealSchedule", type=str, default="Linear", choices=["Linear", "Exp", "Frac"], help="type of anneal schedule")
parser.add_argument("--project_name", type=str, default="")

parser.add_argument("--minib_time_steps", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_schedule", type=str, choices = ["cosine", "const"], default = "cosine")
parser.add_argument("--Energy_lr", type=float, default=0.0)
parser.add_argument("--SDE_lr", type=float, default=10**-5)
parser.add_argument("--SDE_weight_decay", type=float, default=0.)
parser.add_argument("--learn_beta_mode", type=str, default="None", choices=["min_and_max", "max", "None"], help="learn beta min and max, lin interp in-between")

parser.add_argument("--learn_covar", type=bool, default=False, help="learn additional covar of target")
parser.add_argument("--sigma_init", type=float, default=1., help="init value of sigma")
parser.add_argument("--repulsion_strength", type=float, default=0., help="repulsion_strength >= 0")

parser.add_argument("--disable_jit", type=bool, default=False, help="learn additional covar of target")

parser.add_argument("--N_anneal", type=int, default=1000)
parser.add_argument("--N_warmup", type=int, default=0)
parser.add_argument("--steps_per_epoch", type=int, default=10)

parser.add_argument("--update_params_mode", type=str, choices = ["all_in_one", "DKL"], default="all_in_one")
parser.add_argument("--epochs_per_eval", type=int, default=50)

parser.add_argument("--beta_min", type=float, default=0.05)
parser.add_argument("--beta_max", type=float, default=5.)
parser.add_argument('--temp_mode', action='store_true', default=True, help='only for discrete time model')
parser.add_argument('--no-temp_mode', action='store_false', dest='temp_mode', help='')

parser.add_argument("--feature_dim", type=int, default=124)
parser.add_argument("--n_hidden", type=int, default=124)
parser.add_argument("--n_layers", type=int, default=3)

parser.add_argument('--use_interpol_gradient', action='store_true', default=True, help='gradient of energy function is added to the score')
parser.add_argument('--no-use_interpol_gradient', dest='use_interpol_gradient', action='store_false', help='gradient of energy function is added not to the score')

parser.add_argument('--use_normal', action='store_true', default=False, help='gradient of energy function is added to the score')
parser.add_argument('--no-use_normal', dest='use_normal', action='store_false', help='gradient of energy function is not added to the score')

parser.add_argument("--SDE_time_mode", type=str, default="Discrete_Time", choices=["Discrete_Time", "Continuous_Time"], help="SDE Time Mode")
parser.add_argument("--Network_Type", type=str, default="FeedForward", choices=["FourierNetwork", "FeedForward", "LSTMNetwork"], help="SDE Time Mode")
parser.add_argument("--model_seed", type=int, default=0, help="Seed used for model init")

#energy function specific args
parser.add_argument("--Pytheus_challenge", type=int, default=1, choices=[0,1,2,3,4,5], help="Pyhteus Chellange Index")
parser.add_argument("--Scaling_factor", type=float, default=1., help="Scaling factor for Energy Functions")
parser.add_argument("--Variances", type=float, default=0.1, help="Variances of Gaussian Mixtures before scalling when means ~Unif([-0.5,0.5])")


args = parser.parse_args()

if(__name__ == "__main__"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    if args.GPU !=-1:                              # GPU -1 means select GPU via env var in command line
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{str(args.GPU)}"

    #disable JIT compilation

    if(args.disable_jit):
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_debug_nans", True)

    # if(args.lr/args.SDE_lr  < 5):
    #     print("Warning: args.lr/args.SDE_lr  < 5, emperically this ratio is too high")

    N_anneal = args.N_anneal
    epochs = N_anneal + args.N_warmup

    Optimizer_Config = {
        "name": "Adam",
        "lr": args.lr,
        "Energy_lr": args.Energy_lr,
        "SDE_lr": args.SDE_lr,
        "learn_beta_mode": args.learn_beta_mode,
        "epochs": epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "epochs_per_eval": args.epochs_per_eval,
        "SDE_weight_decay": args.SDE_weight_decay,
        "lr_schedule": args.lr_schedule,
    }

    Network_Config = {
        "base_name": "Vanilla",
        "name": args.Network_Type,
        "feature_dim": args.feature_dim,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
        "model_seed": args.model_seed,
        "model_mode": "normal"
    }

    if("Discrete_Time_rKL_Loss" in args.SDE_Loss):

        SDE_Type_Config = {
            "name": "DiscreteTime_SDE", 
            "n_diff_steps": args.n_integration_steps,
            "temp_mode": args.temp_mode,
            "n_integration_steps": args.n_integration_steps,
            "SDE_weightening": args.SDE_weightening,
            "use_normal": False
        }
        
        SDE_Loss_Config = {
            "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
            "SDE_Type_Config": SDE_Type_Config,
            "batch_size": args.batch_size,
            "n_integration_steps": args.n_integration_steps,
            "minib_time_steps": args.minib_time_steps,
        }
    else:
        SDE_Type_Config = {
            "name": args.SDE_Type,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
            "use_interpol_gradient": args.use_interpol_gradient,
            "n_integration_steps": args.n_integration_steps,
            "SDE_weightening": args.SDE_weightening,
            "use_normal": args.use_normal,
            "learn_covar": args.learn_covar,
            "sigma_init": args.sigma_init,
            "repulsion_strength": args.repulsion_strength,
        }
        
        SDE_Loss_Config = {
            "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
            "SDE_Type_Config": SDE_Type_Config,
            "batch_size": args.batch_size,
            "n_integration_steps": args.n_integration_steps,
            "minib_time_steps": args.minib_time_steps,
            "update_params_mode": args.update_params_mode,
            
        }

    n_eval_samples = 2000
    ### TODO implement different scales
    if(args.Energy_Config == "GaussianMixtureToy"):
        torch.manual_seed(0)
        #np.random.seed(42)
        dim = 2
        num_gaussians = 1
        x_min = -1
        x_max = 1
        loc_scaling = 1
        log_var_scaling = 0.1

        mean = (torch.rand((num_gaussians, dim)) - 0.5)*2 * loc_scaling
        log_var = torch.ones((num_gaussians, dim)) * log_var_scaling

        #rand_func = lambda x: np.random.uniform(x_min, x_max, 2)
        Energy_Config = {
            "name": "GaussianMixture",
            "dim_x": 2,
            "means": mean,
            "variances": np.exp(log_var),
            "weights": [1/num_gaussians for i in range(num_gaussians)],
            
        }
    elif(args.Energy_Config == "GaussianMixture"):
        n_eval_samples = 10000
        torch.manual_seed(0)
        #np.random.seed(42)
        dim = 2
        num_gaussians = 40

        loc_scaling = 40
        log_var_scaling = 0.1
        mean = (torch.rand((num_gaussians, dim)) - 0.5)*2*loc_scaling
        log_var = torch.ones((num_gaussians, dim)) * log_var_scaling
        Energy_Config = {
            "name": "GaussianMixture",
            "dim_x": 2,
            "means": mean,
            "variances": torch.exp(log_var),
            "weights": [1/num_gaussians for i in range(num_gaussians)],
            "num_modes": num_gaussians
        }
    elif(args.Energy_Config == "Rastrigin"):
        Energy_Config = {
            "name": "Rastrigin",
            "dim_x": 2,
            "shift": 5.0
        }
    elif(args.Energy_Config == "MexicanHat"):
        Energy_Config = {
            "name": "MexicanHat",
            "dim_x": 2,
        }
    elif(args.Energy_Config == "Pytheus"):
        n_eval_samples = 100
        Energy_Config = {
            "name": "Pytheus",
            "challenge_index": args.Pytheus_challenge,
        }
        n_eval_samples = 10000

    elif("LennardJones" in args.Energy_Config):
        n_eval_samples = 1000
        Network_Config["base_name"] = "EGNN"
        N = 13
        out_dim = 3
        Network_Config["n_particles"] = N
        Network_Config["out_dim"] = out_dim 
        Energy_Config = {
            "name": args.Energy_Config,
            "N": N,
            "dim_x": N*out_dim,
        }
    elif("DoubleWell_iter" in args.Energy_Config):
        Network_Config["base_name"] = "EGNN"
        N = 4
        out_dim = 2
        Network_Config["n_particles"] = N
        Network_Config["out_dim"] = out_dim 
        Energy_Config = {
            "name": args.Energy_Config,
            "N": N,
            "dim_x": N*out_dim,
        }
    elif("DoubleWell_Richter" in args.Energy_Config):
        N = 5
        Energy_Config = {
            "name": args.Energy_Config,
            "d": N,
            "m": N,
            "dim_x": N + N,
        }

    elif("WavePINN" in args.Energy_Config):
        Energy_Config = {
            "name": args.Energy_Config,
            "dim_x": 3, ### x dim is here the latent dim
            "d_in": 1,
            "l1_d": 64,
            "l2_d": 64,
            "d_out": 1,
        }
        n_eval_samples = 10
    elif("DoubleMoon" in args.Energy_Config):
        Energy_Config = {
            "name": args.Energy_Config,
            "d_in": 1,
            "l1_d": 64,
            "l2_d": 64,
            "d_out": 1,
        }
        n_eval_samples = 10
    
    else:
        raise ValueError("Energy Config not found")
    Energy_Config["scaling"] = args.Scaling_factor

    Anneal_Config = {
        "name": args.AnnealSchedule,
        "T_start": args.T_start,
        "T_end": args.T_end,
        "N_anneal": args.N_anneal,
        "N_warmup": args.N_warmup,
        "lam": 10.
    }


    base_config = {
        "EnergyConfig": Energy_Config,
        "Anneal_Config": Anneal_Config,
        "SDE_Loss_Config": SDE_Loss_Config,
        "Optimizer_Config": Optimizer_Config,
        "Network_Config": Network_Config,

        "num_epochs": epochs,
        "n_eval_samples": n_eval_samples,
        "project_name": args.project_name
        
    }

    trainer = TrainerClass(base_config)
    trainer.train()
