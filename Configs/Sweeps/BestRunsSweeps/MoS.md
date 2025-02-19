# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss Bridge_ --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --SDE_lr YY Interpol_lr = 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50


### TODO look for good hyperparameter in new runs sweep_LD_and_LV and sweep_all_frozen

### LD
Interpol_lr = 0.001
lr = 0.0001
beta_max = 2.5
sigma_init = 15

ELBO = 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50

### LV
Interpol_lr = 0.001
lr = 0.0001
beta_max = 2.5
sigma_init = 15.

ELBO = 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50

### LD Frozen
Interpol_lr = 0.001
lr = 0.0005
beta_max = 2,5
sigma_init = 15
learn_SDE_params_mode:
value: prior_only

ELBO = 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --SDE_lr 0.0005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --learn_SDE_params_mode prior_only

### rKL frozen Diverged
lr = 
beta_max = 
sigma_init = 
learn_SDE_params_mode:
value: prior_only

ELBO = 


### LV Frozen
Interpol_lr = 0.001
lr = 0.0005
beta_max = 2.5
sigma_init = 15
learn_SDE_params_mode:
value: prior_only

ELBO = 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --SDE_lr 0.0005 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --learn_SDE_params_mode prior_only



