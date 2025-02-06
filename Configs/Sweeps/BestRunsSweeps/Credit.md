# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config Sonar --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0

### LD
lr = 0.005
beta_max = 0.01
sigma_init = 0.5

ELBO = -504.75

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2

### LV Diverges
best checkpoint taken
lr = 0.002
beta_max = 0.01
sigma_init = 0.5

ELBO = -700

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.002 --SDE_lr 0.002 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2

### LD Frozen
lr = 0.005
beta_max = 0.01
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

ELBO = -505.1

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only

### rKL frozen
lr = 0.005
beta_max = 0.01
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

ELBO = -505.18

python main.py --SDE_Loss Bridge_rKL --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only


### LV Frozen
lr = 0.002
beta_max = 0.01
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

ELBO = -505.21

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.002 --SDE_lr 0.002 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only



