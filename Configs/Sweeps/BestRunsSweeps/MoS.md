# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config Sonar --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0


### TODO look for good hyperparameter in new runs sweep_LD_and_LV and sweep_all_frozen

### LD
lr = 
beta_max = 
sigma_init = 

ELBO = 

### LV Diverged
lr = 
beta_max = 
sigma_init = 

ELBO = 

### LD Frozen
lr = 
beta_max = 
sigma_init = 
learn_SDE_params_mode:
value: prior_only

ELBO = 

### rKL frozen
lr = 
beta_max = 
sigma_init = 
learn_SDE_params_mode:
value: prior_only

ELBO = 


### LV Frozen
lr = 
beta_max = 
sigma_init = 
learn_SDE_params_mode:
value: prior_only

ELBO = 



