import argparse
import os
### TODO make seperate run configs for discrete time and continuous time

parser = argparse.ArgumentParser(description="Denoising Diffusion Sampler")
parser.add_argument("--GPU", type=int, default=6, help="GPU id to use")
parser.add_argument("--model_mode", type=str, default="normal", choices = ["normal", "latent"], help="normal training or latent diffusion")
parser.add_argument("--latent_dim", type=int, default=None)

parser.add_argument("--SDE_Loss", type=str, default="LogVariance_Loss", choices=["Reverse_KL_Loss","Reverse_KL_Loss_stop_grad","LogVariance_Loss", "LogVariance_Loss_MC", 
                                                                                 "LogVariance_Loss_with_grad", "LogVariance_Loss_weighted", "Reverse_KL_Loss_logderiv",
                                                                                 "Bridge_rKL", "Bridge_LogVarLoss", "Bridge_rKL_logderiv", "Bridge_rKL_logderiv_DiffUCO",
                                                                                "Discrete_Time_rKL_Loss_log_deriv", "Discrete_Time_rKL_Loss_reparam"], help="select loss function")
parser.add_argument("--SDE_Type", type=str, default="VP_SDE", choices=["VP_SDE", "subVP_SDE", "VE_SDE", "Bridge_SDE", "Bridge_SDE_with_bug"], help="select SDE type, subVP_SDE is currently deprecated")
parser.add_argument("--Energy_Config", type=str, default="GaussianMixture", choices=["GaussianMixture", "GaussianMixtureToy", "Rastrigin", "LennardJones", 
                                                                                     "DoubleWellEquivariant", "DoubleWell", "Sonar", "Funnel",
                                                                                      "Pytheus", "WavePINN_latent", "WavePINN_hyperparam", "DoubleMoon",
                                                                                      "Banana", "Brownian", "Lorenz", "Seeds", "Ionosphere", "Sonar", "Funnel", "LGCP", "GermanCredit", "MW54",
                                                                                      "StudentTMixture", "FunnelDistrax"], help="EnergyClass")
parser.add_argument("--n_particles", type=int, default=2, help="the dimension can be controlled for some problems")
parser.add_argument("--T_start", type=float, default=[1.], nargs="+" ,  help="Starting Temperature")
parser.add_argument("--T_end", type=float, default=0., help="End Temperature")
parser.add_argument("--anneal_lam", type=float, default=10., help="Strech anneal schedule; not possible for linear schedule")
parser.add_argument("--n_integration_steps", type=int, default=100)
parser.add_argument("--SDE_weightening", type=str, default="normal", choices=["normal", "weighted"], help="SDE weightening")
parser.add_argument("--AnnealSchedule", type=str, default="Linear", choices=["Linear", "Exp", "Frac"], help="type of anneal schedule")
parser.add_argument("--project_name", type=str, default="")

parser.add_argument("--minib_time_steps", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument( "--lr", type=float, default=[0.001], nargs="+")
parser.add_argument("--lr_schedule", type=str, choices = ["cosine", "const", "cosine_warmup"], default = "cosine")
parser.add_argument("--Energy_lr", type=float, default=0.0)
parser.add_argument("--Interpol_lr", type=float, default=0.001)
parser.add_argument("--SDE_lr", type=float, default=[0.001], nargs="+")
parser.add_argument("--SDE_weight_decay", type=float, default=0.)
parser.add_argument("--clip_value", type=float, default=1., help = "clip value of sde param gradients")
parser.add_argument("--learn_SDE_params_mode", type=str, default="all", choices=["prior_only", "all", "all_and_beta"], 
                    help="prior_only: only learn prior params are learned, all: learn all SDE params except betas, all_and_beta: learn all params including beta")

parser.add_argument("--learn_covar", action='store_true', default=False, help="learn additional covar of target")
parser.add_argument("--sigma_init", type=float, default=1., help="init value of sigma")
parser.add_argument("--repulsion_strength", type=float, default=0., help="repulsion_strength >= 0")

### TODO explain the effect
parser.add_argument('--use_off_policy', action='store_true', default=False, help='use off policy sampling')
parser.add_argument('--no-use_off_policy', dest='use_off_policy', action='store_false', help='dont use off policy sampling')
parser.add_argument("--sigma_scale_factor", type=float, default=1., help="amount of noise for off policy sampling, 0 has no effect = no-use_off_policy")

parser.add_argument("--disable_jit", action='store_true', default=False, help="disable jit for debugging")

parser.add_argument("--N_anneal", type=int, default=1000)
parser.add_argument("--N_warmup", type=int, default=0)
parser.add_argument("--steps_per_epoch", type=int, default=10)

parser.add_argument("--beta_schedule", type=str, choices = ["constant", "cosine"], default="constant", help="defines the noise schedule for Bridge_SDE")
parser.add_argument("--update_params_mode", type=str, choices = ["all_in_one", "DKL"], default="all_in_one", help="keep all_in_one as default. This is currently not used")
parser.add_argument("--epochs_per_eval", type=int, default=50)

parser.add_argument("--beta_min", type=float, default=0.05)
parser.add_argument("--beta_max", type=float ,default=[0.1], nargs="+" )
parser.add_argument('--temp_mode', action='store_true', default=True, help='only for discrete time model')
parser.add_argument('--no-temp_mode', action='store_false', dest='temp_mode', help='')

parser.add_argument("--feature_dim", type=int, default=124)
parser.add_argument("--n_hidden", type=int, default=124)
parser.add_argument("--n_layers", type=int, default=3)

parser.add_argument('--use_interpol_gradient', action='store_true', default=True, help='use gradient of energy function to parameterize the score')
parser.add_argument('--no-use_interpol_gradient', dest='use_interpol_gradient', action='store_false', help='dont use gradient of energy function to parameterize the score')
### TODO in SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS they use a high lr for that, maybe we should also amke this possible!
parser.add_argument("--learn_interpolation_params", action='store_true', default=True, help="flag which enables learning of interpolation params between pror and target distribution")
parser.add_argument('--no-learn_interpolation_params', dest='learn_interpolation_params', action='store_false', help='flag which enables learning of interpolation params between pror and target distributio')


parser.add_argument('--use_normal', action='store_true', default=False, help='gradient of energy function is added to the score as in Denoising Diffusion Samplers')
parser.add_argument('--no-use_normal', dest='use_normal', action='store_false', help='if false parameterize energy function gradient as in Learning to learn by gradient descent by gradient descent')

parser.add_argument("--SDE_time_mode", type=str, default="Discrete_Time", choices=["Discrete_Time", "Continuous_Time"], help="SDE Time Mode")
parser.add_argument("--Network_Type", type=str, default="FeedForward", choices=["FourierNetwork", "FeedForward", "LSTMNetwork", "ADAMNetwork"], help="SDE Time Mode")

parser.add_argument("--sample_seed", type=int, default=[42], nargs="+", help="Seeds used to obtain target samples")
parser.add_argument("--model_seeds", type = int ,default=[0], nargs="+" , help="Seed used for model and sampling init")
parser.add_argument("--n_eval_samples", type=int, default=2000, help="Number of samples to use for evaluation")


#energy function specific args
parser.add_argument("--Pytheus_challenge", type=int, default=1, choices=[0,1,2,3,4,5], help="Pyhteus Chellange Index")
parser.add_argument("--Scaling_factor", type=float, default=40., help="Scaling factor for Energy Functions")
parser.add_argument("--Variances", type=float, default=1., help="Variances of Gaussian Mixtures before scalling when means ~Unif([-1,1])")
parser.add_argument("--base_net", type=str, default="Vanilla", choices = ["PISgradnet", "Vanilla", "PISNet"], help="Variances of Gaussian Mixtures before scalling when means ~Unif([-1,1])")



args = parser.parse_args()

if(__name__ == "__main__"):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    if args.GPU != -1:                              # GPU -1 means select GPU via env var in command line
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{str(args.GPU)}"


    # importing after set visible devices seems to be important! othervice all gpus remain visible
    import jax
    from jax import config as jax_config
    import torch
    from Trainer.train import TrainerClass
    import numpy as np

    devices = jax.local_devices()
    print(devices)
    #disable JIT compilation
    #jax.config.update("jax_enable_x64", True)
    if(args.disable_jit):
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_debug_nans", True)

    # if(args.lr/args.SDE_lr  < 5):
    #     print("Warning: args.lr/args.SDE_lr  < 5, emperically this ratio is too high")
    zipped_lr_list = zip(args.lr, args.SDE_lr)
    temp_list = args.T_start
    seed_list = zip(args.model_seeds, args.sample_seed)
    beta_list = args.beta_max

    for beta_max in beta_list:
        for seed, sample_seed in seed_list:
            for temp_start in temp_list:
                for lr, SDE_lr in zipped_lr_list:
                        
                    N_anneal = args.N_anneal
                    epochs = N_anneal + args.N_warmup

                    Optimizer_Config = {
                        "name": "Adam",
                        "lr": lr,
                        "Interpol_lr": args.Interpol_lr,
                        "SDE_lr": SDE_lr,
                        "learn_SDE_params_mode": args.learn_SDE_params_mode,
                        "epochs": epochs,
                        "steps_per_epoch": args.steps_per_epoch,
                        "epochs_per_eval": args.epochs_per_eval,
                        "SDE_weight_decay": args.SDE_weight_decay,
                        "clip_value": args.clip_value,
                        "lr_schedule": args.lr_schedule,
                    }

                    Network_Config = {
                        "base_name": args.base_net,
                        "name": args.Network_Type,
                        "feature_dim": args.feature_dim,
                        "n_hidden": args.n_hidden,
                        "n_layers": args.n_layers,
                        "model_seed": seed,
                        "model_mode": args.model_mode
                    }

                    if("Discrete_Time_rKL_Loss" in args.SDE_Loss):

                        SDE_Type_Config = {
                            "name": "DiscreteTime_SDE", 
                            "n_diff_steps": args.n_integration_steps,
                            "temp_mode": args.temp_mode,
                            "n_integration_steps": args.n_integration_steps,
                            "SDE_weightening": args.SDE_weightening,
                            "use_normal": False,
                        }
                        
                        SDE_Loss_Config = {
                            "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
                            "SDE_Type_Config": SDE_Type_Config,
                            "batch_size": args.batch_size,
                            "n_integration_steps": args.n_integration_steps,
                            "minib_time_steps": args.minib_time_steps,
                    }
                else:
                    #modified sampling distributions are only applicable for certain losses
                    if(args.use_off_policy and (args.SDE_Loss != "LogVariance_Loss" and args.SDE_Loss != "Bridge_LogVarLoss" and args.SDE_Loss != "Reverse_KL_Loss_logderiv" and args.SDE_Loss != "Bridge_rKL_logderiv")):
                        raise ValueError("Off policy only implemented for LogVariance_Loss")
                    if(not args.use_off_policy and args.sigma_scale_factor != 1.):
                        raise ValueError("Sigma scale factor != 0 and use_off_policy is off")
                    if(args.beta_min > beta_max):
                        raise ValueError("Beta min >= beta max")

                    SDE_Type_Config = {
                        "name": args.SDE_Type,
                        "beta_min": args.beta_min,
                        "beta_max": beta_max,
                        "use_interpol_gradient": args.use_interpol_gradient,
                        "n_integration_steps": args.n_integration_steps,
                        "SDE_weightening": args.SDE_weightening,
                        "use_normal": args.use_normal,
                        "learn_covar": args.learn_covar,
                        "sigma_init": args.sigma_init,
                        "repulsion_strength": args.repulsion_strength,
                        "sigma_scale_factor": args.sigma_scale_factor,
                        "batch_size": args.batch_size,
                        "use_off_policy": args.use_off_policy,
                        "learn_interpolation_params": args.learn_interpolation_params,
                        "beta_schedule": args.beta_schedule
                    }

                    SDE_Loss_Config = {
                        "name": args.SDE_Loss, # Reverse_KL_Loss, LogVariance_Loss
                        "SDE_Type_Config": SDE_Type_Config,
                        "batch_size": args.batch_size,
                        "n_integration_steps": args.n_integration_steps,
                        "minib_time_steps": args.minib_time_steps,
                        "update_params_mode": args.update_params_mode,

                    }
                    n_eval_samples = args.n_eval_samples
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
                        torch.manual_seed(seed)
                        #np.random.seed(42)
                        dim = args.n_particles
                        num_gaussians = 40

                        loc_scaling = args.Scaling_factor
                        var_scaling = args.Variances
                        mean = (torch.rand((num_gaussians, dim)) - 0.5)*2*loc_scaling
                        variances = torch.ones((num_gaussians, dim)) * var_scaling
                        Energy_Config = {
                            "name": "GaussianMixture",
                            "dim_x": dim,
                            "means": mean,
                            "variances": variances,#torch.nn.functional.softplus(log_var),
                            "weights": [1/num_gaussians for i in range(num_gaussians)],
                            "num_modes": num_gaussians
                        }
                    elif(args.Energy_Config == "Rastrigin"):
                        dim = args.n_particles
                        Energy_Config = {
                            "name": "Rastrigin",
                            "dim_x": dim,
                            "shift": 0.0
                        }
                    elif(args.Energy_Config == "Pytheus"):
                        Energy_Config = {
                            "name": "Pytheus",
                            "challenge_index": args.Pytheus_challenge,
                        }

                    elif("LennardJones" in args.Energy_Config):
                        Network_Config["base_name"] = "EGNN"
                        N = args.n_particles
                        out_dim = 3
                        Network_Config["n_particles"] = N
                        Network_Config["out_dim"] = out_dim 
                        Energy_Config = {
                            "name": args.Energy_Config,
                            "N": N,
                            "dim_x": N*out_dim,
                        }
                    elif("DoubleWellEquivariant" in args.Energy_Config):
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
                    elif("DoubleWell" in args.Energy_Config):
                        N = args.n_particles
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
                    elif("DoubleMoon" in args.Energy_Config):
                        Energy_Config = {
                            "name": args.Energy_Config,
                            "d_in": 1,
                            "l1_d": 64,
                            "l2_d": 64,
                            "d_out": 1,
                        }
                    elif("Banana" in args.Energy_Config or "Brownian" in args.Energy_Config or "Lorenz" in args.Energy_Config):
                        from EnergyFunctions.EnergyData.BrownianData import load_model_gym
                        _, dim = load_model_gym(args.Energy_Config)
                        Energy_Config = {
                            "name": args.Energy_Config,
                            "dim_x": dim
                        }
                    elif("Seeds" in args.Energy_Config or "Ionosphere" in args.Energy_Config or "Sonar" in args.Energy_Config):
                        from EnergyFunctions.EnergyData.SeedsData import load_model_other
                        _, dim = load_model_other(args.Energy_Config)
                        Energy_Config = {
                            "name": args.Energy_Config,
                            "dim_x": dim
                        }
                    elif(args.Energy_Config == "Funnel"):
                        dim = args.n_particles
                        Energy_Config = {
                            "name": "Funnel",
                            "dim_x": dim,
                            "eta": 3,
                            "scaling": args.Scaling_factor
                        }

                    elif(args.Energy_Config == "LGCP"):
                        Energy_Config = {
                            "name": "LGCP",  # Your 2D array of point coordinates
                            "num_grid_per_dim": 40,      # Grid size (40x40=1600)
                            "use_whitened": False,       # Whether to use whitened parameterization
                            "dim_x": 1600,              # Total dimension (grid_size^2)
                            "scaling": 1.0              # Required by base class
                        }

                    elif(args.Energy_Config == "GermanCredit"):
                        Energy_Config = {
                            "name": "GermanCredit",
                            "dim_x": 25,
                        }
                    elif(args.Energy_Config == "MW54"):
                        N = args.n_particles
                        Energy_Config = {
                            "name": args.Energy_Config,
                            "d": N,
                            "m": N,
                            "dim_x": N + N,
                        }
                    elif(args.Energy_Config == "StudentTMixture"):
                        dim = 50
                        num_components = 10

                        Energy_Config = {
                            "name": "StudentTMixture",
                            "dim_x": dim,
                            "num_components": num_components,
                            "df": 2.0,
                            "seed": seed
                        }

                    else:
                        print(args.Energy_Config)
                        raise ValueError("Energy Config not found")

                    Energy_Config["scaling"] = args.Scaling_factor

                    Network_Config["x_dim"] = Energy_Config["dim_x"]
                    if(Network_Config["model_mode"] == "latent"):
                        SDE_Type_Config["use_interpol_gradient"] = False
                        if(args.latent_dim == None):
                            raise ValueError("Latent dim not defined")


                    Anneal_Config = {
                        "name": args.AnnealSchedule,
                        "T_start": temp_start,
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
                        "project_name": args.project_name,
                        "disable_jit": args.disable_jit,
                        "sample_seed": sample_seed
                    }

                    trainer = TrainerClass(base_config)
                    trainer.train()



