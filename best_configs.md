rkl base
https://wandb.ai/bartmann-jku-linz/DDS_GMM_5d_exploration_laplacian/sweeps/fea6hrk3?nw=nwuserbartmann
elbo -3.6893467903137207
sinkhorn 3083.750732421875
python main.py \
--N_anneal 4000 \
--N_warmup 0 \
--T_end 1 \
--T_start 1 \
--batch_size 2000 \
--beta_max 0.5 \
--beta_min 0.01 \
--Energy_Config GMMDistrax \
--Energy_lr 0 \
--seed 0 \
--epochs_per_eval 50 \
--feature_dim 128 \
--GPU -1 \
--Interpol_lr 0 \
--lr 0.005 \
--model_seed 0 \
--n_eval_samples 2000 \
--n_hidden 128 \
--n_integration_steps 128 \
--n_particles 5 \
--base_name Vanilla \
--n_layers 3 \
--Network_Type FeedForward \
--num_epochs 4000 \
--lr_schedule cosine \
--SDE_lr 0.005 \
--project_name "" \
--sample_seed 0 \
--SDE_Loss Bridge_rKL_logderiv \
--SDE_Type Bridge_SDE \
--sigma_init 1

rkl base sigma sweep
https://wandb.ai/bartmann-jku-linz/DDS_GMM_5d_exploration_laplacian/sweeps/fea6hrk3?nw=nwuserbartmann
https://wandb.ai/bartmann-jku-linz/DDS_GMM_5d_exploration_laplacian/runs/ehqsvohj?nw=nwuserbartmann
elbo -1.4374568462371826
sinkhorn 1160.2979736328125

python main.py \
--N_anneal 4000 \
--N_warmup 0 \
--T_end 1 \
--T_start 1 \
--batch_size 2000 \
--beta_max 0.3 \
--beta_min 0.01 \
--Energy_Config GMMDistrax \
--Energy_lr 0 \
--seed 0 \
--epochs_per_eval 50 \
--feature_dim 128 \
--GPU -1 \
--Interpol_lr 0 \
--lr 0.008 \
--model_seed 0 \
--n_eval_samples 2000 \
--n_hidden 128 \
--n_integration_steps 128 \
--n_particles 5 \
--base_name Vanilla \
--n_layers 3 \
--Network_Type FeedForward \
--num_epochs 4000 \
--lr_schedule cosine \
--SDE_lr 0.005 \
--project_name "" \
--sample_seed 0 \
--SDE_Loss Bridge_rKL_logderiv \
--SDE_Type Bridge_SDE \
--sigma_init 20

rkl anneal
https://wandb.ai/bartmann-jku-linz/DDS_GMM_5d_exploration_laplacian/runs/cea715s5?nw=nwuserbartmann
elbo -2.9538357257843018
sinkhorn 146.228589
python main.py \
--N_anneal 4000 \
--N_warmup 0 \
--T_end 1 \
--T_start 100 \
--batch_size 2000 \
--beta_max 1 \
--beta_min 0.01 \
--Energy_Config GMMDistrax \
--Energy_lr 0 \
--seed 0 \
--epochs_per_eval 50 \
--feature_dim 128 \
--GPU -1 \
--Interpol_lr 0 \
--lr 0.005 \
--model_seed 0 \
--n_eval_samples 2000 \
--n_hidden 128 \
--n_integration_steps 128 \
--n_particles 5 \
--base_name Vanilla \
--n_layers 3 \
--Network_Type FeedForward \
--num_epochs 4000 \
--lr_schedule cosine \
--SDE_lr 0.005 \
--project_name "" \
--sample_seed 0 \
--SDE_Loss Bridge_rKL_logderiv \
--SDE_Type Bridge_SDE \
--sigma_init 1


rkl anneal +off policy sigma init 1
https://wandb.ai/bartmann-jku-linz/DDS_GMM_5d_exploration_laplacian/runs/gf6z3ktn?nw=nwuserbartmann
elbo -3.839370
sinkhorn 3069.608252

python main.py \
--N_anneal 4000 \
--N_warmup 0 \
--T_end 1 \
--T_start 1.1 \
--batch_size 2000 \
--beta_max 0.3 \
--beta_min 0.01 \
--Energy_Config GMMDistrax \
--Energy_lr 0 \
--seed 0 \
--epochs_per_eval 50 \
--feature_dim 128 \
--GPU -1 \
--Interpol_lr 0 \
--lr 0.005 \
--model_seed 0 \
--n_eval_samples 2000 \
--n_hidden 128 \
--n_integration_steps 128 \
--n_particles 5 \
--base_name Vanilla \
--n_layers 3 \
--Network_Type FeedForward \
--num_epochs 4000 \
--off_policy_mode laplace \
--lr_schedule cosine \
--SDE_lr 0.005 \
--project_name "" \
--sample_seed 0 \
--SDE_Loss Bridge_rKL_logderiv \
--SDE_Type Bridge_SDE \
--sigma_init 1


rkl off policy sigma init tuned
https://wandb.ai/bartmann-jku-linz/DDS_GMM_10d_exploration_laplacian/runs/lqsj3za6?nw=nwuserbartmann
elbo -1.592895269393921
sinkhorn 1006.8403930664062

python main.py \
--N_anneal 4000 \
--N_warmup 0 \
--T_end 1 \
--T_start 1.1 \
--batch_size 2000 \
--beta_max 0.5 \
--beta_min 0.01 \
--Energy_Config GMMDistrax \
--Energy_lr 0 \
--n_particles 5 \
--seed 0 \
--epochs_per_eval 50 \
--feature_dim 128 \
--GPU 0 \
--Interpol_lr 0 \
--lr 0.005 \
--model_seed 0 \
--n_eval_samples 2000 \
--n_hidden 128 \
--n_integration_steps 128 \
--base_name Vanilla \
--model_mode normal \
--n_layers 3 \
--Network_Type FeedForward \
--off_policy_mode laplace \
--lr_schedule cosine \
--project_name "" \
--sample_seed 0 \
--SDE_Loss Bridge_rKL_logderiv \
--SDE_Type Bridge_SDE \
--sigma_init 20