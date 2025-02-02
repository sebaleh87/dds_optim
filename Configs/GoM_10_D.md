
off-policy
''' 
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1.005 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --n_particles 10 --use_off_policy
''' 
''' 
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1.005 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --Interpol_lr 0.0001 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --n_particles 10 --use_off_policy
''' 

''' 
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1.01 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --n_particles 10 --use_off_policy
''' 


Anneal
''' 
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 8. --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --n_particles 10 
''' 

No explore
''' 
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --n_particles 10 
''' 

LV Loss
''' 
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --Interpol_lr 0.0 --N_anneal 4000 --feature_dim 128 --n_hidden 128 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --n_particles 10
''' 