
### Discrete Diffusion Sampler on Rastragin
python main.py --SDE_Loss Discrete_Time_rKL_Loss_reparam --Energy_Config Rastrigin --n_integration_steps 10 --T_start 9 --batch_size 400 --lr 0.002 --N_anneal 1000 --feature_dim 32 --n_hidden 200 --GPU 6


### COntinuous Diffusion Sampler
This works:
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Rastrigin --n_integration_steps 50 --T_start 9 --batch_size 400 --lr 0.002 --N_anneal 1000 --feature_dim 32 --n_hidden 200 --GPU 6 --beta_max 10.


Does not work:
python main.py --SDE_Loss LogVariance_Loss_MC --Energy_Config Rastrigin --n_integration_steps 50 --T_start 3 --batch_size 400 --lr 0.001 --N_anneal 1000 --feature_dim 32 --n_hidden 200 --GPU 6 --beta_max 5. --minib_time_steps 10




###subVP_SDE
TEST: also works
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Rastrigin --n_integration_steps 50 --T_start 9 --batch_size 400 --lr 0.001 --N_anneal 1000 --feature_dim 32 --n_hidden 200 --GPU 6 --beta_max 5. --SDE_Type subVP_SDE


### test model with gradients 
works!
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Rastrigin --n_integration_steps 50 --T_start 9. --batch_size 400 --lr 0.002 --N_anneal 1000 --feature_dim 32 --n_hidden 200 --GPU 6 --beta_max 10. --use_interpol_gradient


### test model with gradients on pytheus

python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Pytheus --n_integration_steps 50 --T_start 0.2 --batch_size 200 --lr 0.002 --N_anneal 1000 --feature_dim 32 --n_hidden 200 --GPU 5 --beta_max 10. --use_interpol_gradient --Network_Type LSTMNetwork