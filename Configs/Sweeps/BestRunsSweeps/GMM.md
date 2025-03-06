# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 1 2 --sample_seeds 0 1 2 --base_net PISgradnet


### selected according to lowest sinkhorn distance at the end of training


### LD new

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --n_particles 50 --n_eval_samples 16000


### LV new

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --n_eval_samples 16000

### LD Frozen 
Interpol_lr = 0.01
lr = 0.0001
beta_max = 1.
sigma_init = 80.
learn_SDE_params_mode:
value: prior_only

ELBO = 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 50 --n_eval_samples 16000

### rKL frozen
lr = 
beta_max = 
sigma_init = 
learn_SDE_params_mode:
value: prior_only

ELBO = 

python main.py --SDE_Loss Bridge_rKL --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 50 --n_eval_samples 16000

### LV Frozen 
Interpol_lr = 0.001
lr = 0.00005
beta_max = 1.
sigma_init = 80.
learn_SDE_params_mode:
value: prior_only

ELBO = 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 50 --n_eval_samples 16000



### exploration off-policy
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 2. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 40. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --n_eval_samples 16000 --use_off_policy --off_policy_mode laplace --laplace_width 1. --mixture_probs 0.025 --weight_temperature 0


### exploration Anneal
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 200. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 40. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --n_eval_samples 16000 --anneal_lam 10 --AnnealSchedule Exp

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 200. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 40. --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --n_eval_samples 16000 --anneal_lam 20 --AnnealSchedule Exp
