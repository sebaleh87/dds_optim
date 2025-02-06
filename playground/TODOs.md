TODO:

- fix bridge SDE n_eff and Z calculation
- add repulisve energy interpolation
- add possibility do do mcmc over time steps
- sample out of ODE and not out of SDE


-- compare NIS with iterated denoising. would be ablation on mass covering based objective?
##FUNNEL
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --n_particles 10

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.1 --n_particles 10

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 5 --n_particles 10

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2 --n_particles 10

##SONAR
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Sonar --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.

##MW54
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config DoubleWell --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --n_particles 5

##LGCP
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 1000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 1000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.1

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 1000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2.

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 1000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 1000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.

batch size 300
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 300 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 300 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2.