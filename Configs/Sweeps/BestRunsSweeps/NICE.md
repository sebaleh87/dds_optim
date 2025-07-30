# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 1 2 --sample_seeds 0 1 2 --base_net PISgradnet


### selected according to lowest sinkhorn distance at the end of training


###test
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config NICE --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name GradientInspection --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --n_particles 50 --n_eval_samples 2000             
 

### LD new




### LV new



### LV Frozen 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config NICE --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.2 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_07 --use_normal --SDE_Type Bridge_SDE --sigma_init 3. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --learn_SDE_params_mode prior_only --beta_schedule cosine --n_particles 196 --n_eval_samples 16000

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config NICE --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.2 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_07 --use_normal --SDE_Type Bridge_SDE --sigma_init 3. --model_seeds 1 --sample_seed 1 --base_net PISgradnet --learn_SDE_params_mode prior_only --beta_schedule cosine --n_particles 196 --n_eval_samples 16000

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config NICE --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.2 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_07 --use_normal --SDE_Type Bridge_SDE --sigma_init 3. --model_seeds 2 --sample_seed 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --beta_schedule cosine --n_particles 196 --n_eval_samples 16000

### rKL frozen

python main.py --SDE_Loss Bridge_rKL --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_13_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 3 4 5 --sample_seed 3 4 5 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 50 --n_eval_samples 16000

### LV Frozen 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_13_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 3 4 5 6 7 8 9 --sample_seed 3 4 5 6 7 8 9 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 50 --n_eval_samples 16000

