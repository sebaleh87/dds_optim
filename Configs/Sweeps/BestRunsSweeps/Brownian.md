# gridsearch performed over: sigma_init, beta_max, lr

argparse:
python main.py --SDE_Loss ZZ --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0


### TODO look for good hyperparameter in new runs sweep_LD_and_LV and sweep_all_frozen


### LD 
lr = 0.005
beta_max = 0.1
sigma_init = 0.5

ELBO = -0.90

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 7 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2


### LV 
best intermediate result is taken
lr = 0.001
beta_max = 0.05
sigma_init = 0.5

ELBO = 6.12

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2

### LD frozen
lr = 0.005
beta_max = 0.05
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

ELBO = -0.19

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only


### rKL frozen
lr = 0.005
beta_max = 0.05
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

ELBO = 0.48

python main.py --SDE_Loss Bridge_rKL --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only

### LV frozen
lr = 0.002
beta_max = 0.05
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only
learn_SDE_params_mode:
value: prior_only

ELBO = 0.12

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.002 --SDE_lr 0.002 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 7 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only

