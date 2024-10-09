from EnergyFunctions import Rastragin
import jax
from jax import config as jax_config
import os
from Trainer.train import TrainerClass
import argparse

parser = argparse.ArgumentParser(description="Denoising Diffusion Sampler")
parser.add_argument("--gpu", type=str, default="5", help="GPU id to use")
parser.add_argument("--SDE_Loss", type=str, default="LogVariance_Loss", choices=["Reverse_KL_Loss","LogVariance_Loss", "LogVarianceLoss_MC_Class"], help="GPU id to use")
args = parser.parse_args()

if(__name__ == "__main__"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{str(args.gpu)}"
    #disable JIT compilation
    #jax.config.update("jax_disable_jit", True)

    N_anneal = 1000
    epochs = N_anneal
    steps_per_epoch = 10

    Optimizer_Config = {
        "name": "Adam",
        "lr": 1e-3,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
    }

    SDE_Type_Config = {
        "name": "VP_SDE",
        "beta_min": 0.05,
        "beta_max": 5.0,
    }
    
    SDE_Loss_Config = {
        "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
        "SDE_Type_Config": SDE_Type_Config,
        "batch_size": 200,
        "n_integration_steps": 200,
        "minib_time_steps": 100
    }

    Energy_Config = {
        "name": "GaussianMixture",
        "dim_x": 1,
        "means": [-2.0, 2.0, 5.0],
        "variances": [1.0, 1.0, 1.],
        "weights": [0.5, 0.5, 0.5],
    }

    Anneal_Config = {
        "name": "Linear",
        "T_start": 0.2,
        "T_end": 0.0,
        "N_anneal": 1000,
    }

    base_config = {
        "EnergyConfig": Energy_Config,
        "Anneal_Config": Anneal_Config,
        "SDE_Loss_Config": SDE_Loss_Config,
        "Optimizer_Config": Optimizer_Config,

        "n_hidden": 64,
        "n_layers": 3,

        "num_epochs": epochs,
        
    }

    trainer = TrainerClass(base_config)
    trainer.train()