
### LD 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_DBS_02_04 --use_normal --SDE_Type Bridge_SDE --sigma_init 0.2 --model_seeds 0 1 2 --sample_seed 0 1 2 --Bridge_Type DBS --base_net PISgradnet


### LV 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_DBS_02_04 --use_normal --SDE_Type Bridge_SDE --sigma_init 0.2 --model_seeds 0 1 2 --sample_seed 0 1 2 --Bridge_Type DBS --base_net PISgradnet

### LD frozen
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_DBS_02_04 --use_normal --SDE_Type Bridge_SDE --sigma_init 0.2 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only --Bridge_Type DBS --base_net PISgradnet


### rKL frozen
python main.py --SDE_Loss Bridge_rKL --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_DBS_02_04 --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only --Bridge_Type DBS --base_net PISgradnet

### LV frozen
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_DBS_02_04 --use_normal --SDE_Type Bridge_SDE --sigma_init 0.2 --model_seeds 0 1 2 --sample_seed 0 1 2 --learn_SDE_params_mode prior_only --Bridge_Type DBS --base_net PISgradnet

