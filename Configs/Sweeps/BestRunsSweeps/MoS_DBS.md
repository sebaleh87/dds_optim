# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss Bridge_ --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --SDE_lr YY Interpol_lr = 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50


### TODO look for good hyperparameter in new runs sweep_LD_and_LV and sweep_all_frozen

### LD

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.000005 --SDE_lr 0.000005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 7 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS  --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 3 4 5 6 7 8 9 --sample_seed 3 4 5 6 7 8 9 --base_net PISgradnet --n_particles 50 --n_eval_samples 16000 --Bridge_Type DBS

### LV

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.00 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 3 4 5 6 7 8 9  --sample_seed 3 4 5 6 7 8 9  --base_net PISgradnet --n_particles 50 --n_eval_samples 16000 --Bridge_Type DBS

### LD Frozen

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.000005 --SDE_lr 0.000005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 3 4 5 6 7 8 9 --sample_seed 3 4 5 6 7 8 9 --base_net PISgradnet --n_particles 50 --learn_SDE_params_mode prior_only --n_eval_samples 16000 --Bridge_Type DBS


### rKL frozen 

python main.py --SDE_Loss Bridge_rKL --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00001 --SDE_lr 0.00001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 1.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS  --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 0 1 2 --sample_seed 0 1 2 --base_net PISgradnet --n_particles 50 --learn_SDE_params_mode prior_only --n_eval_samples 16000  --Bridge_Type DBS


### LV Frozen

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config StudentTMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 1.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS  --use_normal --SDE_Type Bridge_SDE --sigma_init 15 --model_seeds 3 4 5 6 7 8 9 --sample_seed 3 4 5 6 7 8 9 --base_net PISgradnet --n_particles 50 --learn_SDE_params_mode prior_only --n_eval_samples 16000 --Bridge_Type DBS 





