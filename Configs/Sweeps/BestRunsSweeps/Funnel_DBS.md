# gridsearch performed over: sigma_init, beta_max, lr

python main.py --SDE_Loss ZZ --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr YY --Energy_lr 0.0 --SDE_lr YY --Interpol_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max XX --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns --use_normal --SDE_Type Bridge_SDE --sigma_init XX --model_seeds 0 --n_particles 10

### LD 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --SDE_lr 0.0005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --n_particles 10  --Bridge_Type DBS --base_net PISgradnet


### LV 

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.00005 --SDE_lr 0.00005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 1 --model_seeds 0 1 2  --sample_seed 0 1 2 --n_particles 10  --Bridge_Type DBS --base_net PISgradnet


### LD Frozen

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --SDE_lr 0.0005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2  --sample_seed 0 1 2 --n_particles 10 --learn_SDE_params_mode prior_only  --Bridge_Type DBS --base_net PISgradnet

### rKL Frozen

python main.py --SDE_Loss Bridge_rKL --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --SDE_lr 0.0005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 1 --model_seeds 0 1 2  --sample_seed 0 1 2 --n_particles 10 --learn_SDE_params_mode prior_only  --Bridge_Type DBS --base_net PISgradnet

### LV Frozen

python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Funnel --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_12_03_DBS --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 1 2 --sample_seed 0 1 2 --n_particles 10 --learn_SDE_params_mode prior_only  --Bridge_Type DBS --base_net PISgradnet





