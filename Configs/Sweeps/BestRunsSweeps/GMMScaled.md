
### no-anneal
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistraxScaled --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --n_particles 10 --n_eval_samples 16000

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistraxScaled --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0001 --SDE_lr 0.0001 --Interpol_lr 0.01 --N_anneal 8000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 80. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --n_particles 10 --n_eval_samples 16000


### anneal
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistraxScaled --n_integration_steps 128 --T_start 150 --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --n_particles 10 --n_eval_samples 16000

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GMMDistraxScaled --n_integration_steps 128 --T_start 150 --T_end 1. --batch_size 2000 --lr 0.001 --SDE_lr 0.001 --Interpol_lr 0.001 --N_anneal 8000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.3 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FinalGridRuns_28_03 --use_normal --SDE_Type Bridge_SDE --sigma_init 1. --model_seeds 0 --sample_seed 0 --base_net PISgradnet --n_particles 10 --n_eval_samples 16000
