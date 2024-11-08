from EnergyFunctions import Rastrigin
import jax
from jax import config as jax_config
import os
from Trainer.train import TrainerClass
import argparse
import numpy as np
### TODO make seperate run configs for discrete time and continuous time

parser = argparse.ArgumentParser(description="Denoising Diffusion Sampler")
parser.add_argument("--GPU", type=str, default="5", help="GPU id to use")
parser.add_argument("--SDE_Loss", type=str, default="Reverse_KL_Loss", choices=["Reverse_KL_Loss","LogVariance_Loss", "LogVariance_Loss_MC", "Discrete_Time_rKL_Loss_log_deriv", "Discrete_Time_rKL_Loss_reparam"], help="GPU id to use")
parser.add_argument("--SDE_Type", type=str, default="VP_SDE", choices=["VP_SDE", "subVP_SDE"], help="GPU id to use")
parser.add_argument("--Energy_Config", type=str, default="Rastrigin", choices=["GaussianMixture", "Rastrigin", "MexicanHat", "Pytheus"], help="EnergyClass")
parser.add_argument("--T_start", type=float, default=1., help="Starting Temperature")
parser.add_argument("--T_end", type=float, default=0., help="End Temperature")
parser.add_argument("--n_integration_steps", type=int, default=10)
parser.add_argument("--minib_time_steps", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--N_anneal", type=int, default=1000)
parser.add_argument("--N_warmup", type=int, default=0)
parser.add_argument("--steps_per_epoch", type=int, default=100)


parser.add_argument("--beta_min", type=float, default=1e-6)
parser.add_argument("--beta_max", type=float, default=0.05)
parser.add_argument('--temp_mode', action='store_true', default=False, help='only for discrete time model')
parser.add_argument('--no-temp_mode', action='store_false', help='')

parser.add_argument("--feature_dim", type=int, default=32)
parser.add_argument("--n_hidden", type=int, default=124)
parser.add_argument("--n_layers", type=int, default=3)

parser.add_argument('--use_interpol_gradient', action='store_true', default=True, help='gradient of energy function is added to the score')
parser.add_argument('--no-use_interpol_gradient', action='store_false', help='gradient of energy function is added not to the score')


parser.add_argument("--SDE_time_mode", type=str, default="Discrete_Time", choices=["Discrete_Time", "Continuous_Time"], help="SDE Time Mode")
parser.add_argument("--Network_Type", type=str, default="FeedForward", choices=["FourierNetwork", "FeedForward", "LSTMNetwork"], help="SDE Time Mode")
parser.add_argument("--Pytheus_challenge", type=int, default=1, choices=[0,1,2,3,4,5], help="Pyhteus Chellange Index")
args = parser.parse_args()

if(__name__ == "__main__"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    if args.GPU !=-1:                              # GPU -1 means select GPU via env var in command line
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{str(args.GPU)}"
    #disable JIT compilation
    #jax.config.update("jax_disable_jit", True)

    N_anneal = args.N_anneal
    epochs = N_anneal + args.N_warmup

    Optimizer_Config = {
        "name": "Adam",
        "lr": args.lr,
        "epochs": epochs,
        "steps_per_epoch": args.steps_per_epoch,
    }

    Network_Config = {
        "name": args.Network_Type,
        "feature_dim": args.feature_dim,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
    }

    if("Discrete_Time_rKL_Loss" in args.SDE_Loss):

        SDE_Type_Config = {
            "name": "DiscreteTime_SDE", 
            "n_diff_steps": args.n_integration_steps,
            "temp_mode": args.temp_mode,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
        }
        
        SDE_Loss_Config = {
            "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
            "SDE_Type_Config": SDE_Type_Config,
            "batch_size": args.batch_size,
            "n_integration_steps": args.n_integration_steps,
            "minib_time_steps": args.minib_time_steps
        }
    else:
        SDE_Type_Config = {
            "name": "VP_SDE", 
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
            "use_interpol_gradient": args.use_interpol_gradient,
        }
        
        SDE_Loss_Config = {
            "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
            "SDE_Type_Config": SDE_Type_Config,
            "batch_size": args.batch_size,
            "n_integration_steps": args.n_integration_steps,
            "minib_time_steps": args.minib_time_steps
        }

    if(args.Energy_Config == "GaussianMixture"):
        np.random.seed(42)
        num_gaussians = 40
        x_min = -40
        x_max = 40
        rand_func = lambda x: np.random.uniform(x_min, x_max, 2)
        Energy_Config = {
            "name": "GaussianMixture",
            "dim_x": 2,
            "means": [rand_func(i) for i in range(num_gaussians)],
            "variances": [[1.10,1.10] for i in range(num_gaussians)],
            "weights": [1/num_gaussians for i in range(num_gaussians)],
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
        Energy_Config = {
            "name": "Pytheus",
            "challenge_index": args.Pytheus_challenge,
        }
    else:
        raise ValueError("Energy Config not found")

    Anneal_Config = {
        "name": "Linear",
        "T_start": args.T_start,
        "T_end": 0.0,
        "N_anneal": args.N_anneal,
        "N_warmup": args.N_warmup,
    }


    base_config = {
        "EnergyConfig": Energy_Config,
        "Anneal_Config": Anneal_Config,
        "SDE_Loss_Config": SDE_Loss_Config,
        "Optimizer_Config": Optimizer_Config,
        "Network_Config": Network_Config,

        "num_epochs": epochs,
        "n_eval_samples": 10*1000
        
    }

    trainer = TrainerClass(base_config)
    trainer.train()