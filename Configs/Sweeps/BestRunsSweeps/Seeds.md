# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 1 2 --sample_seed 0 1 2

### LD 
lr = 0.005
beta_max = 0.1
sigma_init = 0.5

ELBO = -73.49

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2

### LV
lr = 0.005
beta_max = 0.1
sigma_init = 0.5

ELBO = -73.56

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2


### LD Frozen
lr = 0.005
beta_max = 0.1
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

ELBO = -74.19

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only


### LV Frozen
lr = 0.005
beta_max = 0.1
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

ELBO = -74.2

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only


### rKL Frozen
lr = 0.002
beta_max = 0.1
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

ELBO = -74.35

python main.py --SDE_Loss Bridge_rKL --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.002 --SDE_lr 0.002 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only



