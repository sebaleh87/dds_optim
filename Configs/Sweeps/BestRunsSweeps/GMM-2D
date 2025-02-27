
GMM no langevin precond vs langevin precond:

## lV
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --Energy_lr 0.0 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name langevin_precon --use_normal --SDE_Type Bridge_SDE  --sigma_init 40. --model_seed 0 --n_particles 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --langevin_precon True


## LV no langevin
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --Energy_lr 0.0 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 1 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name langevin_precon --use_normal --SDE_Type Bridge_SDE  --sigma_init 40. --model_seed 0 --n_particles 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --langevin_precon False

## rKL_LD
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --Energy_lr 0.0 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 2 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name langevin_precon --use_normal --SDE_Type Bridge_SDE  --sigma_init 40. --model_seed 0 --n_particles 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --langevin_precon True


## rKL_LD no langevin
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistrax --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --Energy_lr 0.0 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 3 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name langevin_precon --use_normal --SDE_Type Bridge_SDE  --sigma_init 40. --model_seed 0 --n_particles 2 --base_net PISgradnet --learn_SDE_params_mode prior_only --langevin_precon False