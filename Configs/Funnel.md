
project name = DDS_Funnel_30_01_2025_17_04_dim_10
w2 distanz circa 100, elbo circa -0.2
'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.06 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name 30_01_2025_17_04 --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2. --n_particles 10 --Interpol_lr 0.01 --model_seeds 0 --sample_seed 42 --model_seed 0 --sampling_seed 0
'''


python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.06 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name 30_01_2025_17_04 --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2. --n_particles 10 --Interpol_lr 0.01 --model_seeds 1 --sample_seed 1


python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.06 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name 30_01_2025_17_04 --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2. --n_particles 10 --Interpol_lr 0.01 --model_seeds 2 --sample_seed 2 

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.06 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name 30_01_2025_17_04 --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2. --n_particles 10 --Interpol_lr 0.01 --model_seeds 3 --sample_seed 3 