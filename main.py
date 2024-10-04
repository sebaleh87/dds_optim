from EnergyFunctions import Rastragin
import jax
from jax import config as jax_config
import os
from Trainer.train import TrainerClass

if(__name__ == "__main__"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    #disable JIT compilation
    #jax.config.update("jax_disable_jit", True)

    SDE_Type_Config = {
        "name": "VP_SDE",
        "beta_min": 0.1,
        "beta_max": 10.0,
    }
    
    SDE_Loss_Config = {
        "name": "Reverse_KL_Loss",
        "SDE_Type_Config": SDE_Type_Config,
    }

    Energy_Config = {
        "name": "GaussianMixture",
        "dim_x": 1,
        "means": [-5.0, 5.0],
        "variances": [1.0, 1.0],
        "weights": [0.5, 0.5],
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

        "lr": 1e-3,
        "n_hidden": 64,
        "n_layers": 3,
        "num_epochs": Anneal_Config["N_anneal"],
        "steps_per_epoch": 10,
        "batch_size": 100,
        "n_integration_steps": 200,
    }

    trainer = TrainerClass(base_config)
    trainer.train()