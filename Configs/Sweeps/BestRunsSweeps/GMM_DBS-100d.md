
### LD Frozen 


python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 3  --sample_seed 0 1 2 3 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 100 --n_eval_samples 16000 --Bridge_Type DBS

### rKL frozen
python main.py --SDE_Loss Bridge_rKL --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 3 --sample_seed 0 1 2 3 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 100 --n_eval_samples 16000 --Bridge_Type DBS

### LV Frozen 


python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 3 --sample_seed 0 1 2 3 --base_net PISgradnet --learn_SDE_params_mode prior_only --n_particles 100 --n_eval_samples 16000 --Bridge_Type DBS


### LD 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 3 4 --sample_seed 0 1 2 3 4 --base_net PISgradnet --n_particles 100 --n_eval_samples 16000 --Bridge_Type DBS


### LV 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 2.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_03_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 1 2 3 4 --sample_seed 0 1 2 3 4 --base_net PISgradnet --n_particles 100 --n_eval_samples 16000 --Bridge_Type DBS

