from EnergyFunctions import Rastrigin
import jax
from jax import config as jax_config
import os
from Trainer.train import TrainerClass
import argparse

### TODO make seperate run configs for discrete time and continuous time

parser = argparse.ArgumentParser(description="Denoising Diffusion Sampler")
parser.add_argument("--gpu", type=str, default="5", help="GPU id to use")
parser.add_argument("--SDE_Loss", type=str, default="Discrete_Time_rKL_Loss", choices=["Reverse_KL_Loss","LogVariance_Loss", "LogVarianceLoss_MC_Class", "Discrete_Time_rKL_Loss"], help="GPU id to use")
parser.add_argument("--Energy_Config", type=str, default="GaussianMixture", choices=["GaussianMixture", "Rastrigin", "MexicanHat"], help="EnergyClass")
parser.add_argument("--T_start", type=float, default=2., help="Starting Temperature")
parser.add_argument("--T_end", type=float, default=0., help="End Temperature")
parser.add_argument("--n_integration_steps", type=int, default=200)
parser.add_argument("--minib_time_steps", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--N_anneal", type=int, default=1000)
parser.add_argument("--N_warmup", type=int, default=0)
parser.add_argument("--SDE_time_mode", type=str, default="Discrete_Time", choices=["Discrete_Time", "Continuous_Time"], help="SDE Time Mode")
args = parser.parse_args()

if(__name__ == "__main__"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{str(args.gpu)}"
    #disable JIT compilation
    #jax.config.update("jax_disable_jit", True)

    N_anneal = args.N_anneal
    epochs = N_anneal + args.N_warmup
    steps_per_epoch = 10

    Optimizer_Config = {
        "name": "Adam",
        "lr": args.lr,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
    }

    Network_Config = {
        "name": "FourierNetwork",
        "feature_dim": 64,
        "n_hidden": 120,
        "n_layers": 3,
    }

    if(args.SDE_Loss == "Discrete_Time_rKL_Loss"):
        SDE_Type_Config = {
            "name": "DiscreteTime_SDE", 
            "n_diff_steps": args.n_integration_steps,
        }
        
        SDE_Loss_Config = {
            "name": "Discrete_Time_rKL_Loss", # Reverse_KL_Loss, LogVariance_Loss
            "SDE_Type_Config": SDE_Type_Config,
            "batch_size": args.batch_size,
            "n_integration_steps": args.n_integration_steps,
            "minib_time_steps": args.minib_time_steps
        }
    else:
        SDE_Type_Config = {
            "name": "VP_SDE", 
            "beta_min": 0.05,
            "beta_max": 5.0,
        }
        
        SDE_Loss_Config = {
            "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
            "SDE_Type_Config": SDE_Type_Config,
            "batch_size": args.batch_size,
            "n_integration_steps": args.n_integration_steps,
            "minib_time_steps": args.minib_time_steps
        }

    if(args.Energy_Config == "GaussianMixture"):
        Energy_Config = {
            "name": "GaussianMixture",
            "dim_x": 1,
            "means": [-2.0, 2.0, 5.0],
            "variances": [1.0, 1.0, 1.],
            "weights": [0.5, 0.5, 0.5],
        }
    elif(args.Energy_Config == "Rastrigin"):
        Energy_Config = {
            "name": "Rastrigin",
            "dim_x": 2,
        }
    elif(args.Energy_Config == "MexicanHat"):
        Energy_Config = {
            "name": "MexicanHat",
            "dim_x": 2,
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
        
    }

    trainer = TrainerClass(base_config)
    trainer.train()