# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 300 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 1 2 --sample_seed 0 1 2

### LD 
lr = 0.005
beta_max = 0.3
sigma_init = 1.

ELBO = 459.48

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 300 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2

### LV
lr = 0.005
beta_max = 0.1
sigma_init = 1.

ELBO = 468.37

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config LGCP --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 300 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2

### LD Frozen
lr = 0.005
beta_max = 0.3
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

ELBO = -460.21

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 300 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only


### LV Frozen
lr = 0.005
beta_max = 0.1 
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

ELBO = -468.69

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config LGCP --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 300 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only

### rKL Frozen
lr = 0.005
beta_max = 0.3
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

ELBO = -467.388

python main.py --SDE_Loss Bridge_rKL --Energy_Config LGCP --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 300 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only




