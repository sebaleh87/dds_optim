# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 --n_particles 10

### LD 
lr = 0.005
beta_max = 0.3
sigma_init = 1.

Sinkhorn = 102.05

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --n_particles 10

## LD beta_schedule
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 --n_particles 10 --beta_schedule learned


### LV 
lr = 0.002
beta_max = 0.1
sigma_init = 0.5

Sinkhorn = 114.41

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.002 --SDE_lr 0.002 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 0.5 --model_seeds 0 1 2 --n_particles 10


### LD Frozen
lr = 0.005
beta_max = 0.3
sigma_init = 0.5
learn_SDE_params_mode:
value: prior_only

Sinkhorn = 100.7

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --n_particles 10 --learn_SDE_params_mode prior_only

### rKL Frozen
lr = 0.001
beta_max = 0.3
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

Sinkhorn = 114.33

python main.py --SDE_Loss Bridge_rKL --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --n_particles 10 --learn_SDE_params_mode prior_only

### LV Frozen
lr = 0.002
beta_max = 0.3
sigma_init = 1.
learn_SDE_params_mode:
value: prior_only

Sinkhorn = 102.57

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.002 --SDE_lr 0.002 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 1 2 3 --n_particles 10 --learn_SDE_params_mode prior_only





