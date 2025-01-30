DDS_GermanCredit_29_01_2025_17_04_dim_25
'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name  29_01_2025_17_04 --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 1 2
'''



frozen:

'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.05 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name paper_runs_frozen --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 --learn_SDE_params_mode prior_only
'''