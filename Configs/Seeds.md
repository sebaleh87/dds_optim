
### paper runs DDS_Seeds_paper_runs_dim_26

checkpoints = ["iconic-butterfly-5", "radiant-moon-4", "clean-bird-3"]
'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name paper_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.3 --model_seed 0 1 2 
''' 


frozen: ELBO -73.34

'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name paper_runs_frozen --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.3 --model_seed 0 --learn_SDE_params_mode prior_only     
''' 


Ablation configs:
configs for ablation on sigma with and without learning

'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.1 0.2 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name SigmaAblation --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.3 --model_seed 0 --learn_SDE_params_mode prior_only  
'''

'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.1 0.2 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name SigmaAblation --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.3 --model_seed 0
'''


off policy:

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1.005 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name paper_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.3 --model_seed 0 1 2 --use_off_policy