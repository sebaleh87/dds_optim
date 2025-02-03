
wandb_ids = ["resilient-wind-3", "fanciful-leaf-2", "wandering-fog-1"]
project name = DDS_LGCP_29_01_2025_17_04_dim_1600
'''
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 300 --lr 0.005 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 90 --GPU 0 --beta_max 0.5 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name 29_01_2025_17_04 --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --clip_value 1. --model_seed 0 1 2
'''

