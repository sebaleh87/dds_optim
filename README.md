#Installation

```
conda env create -f environment.yml
```

When installation crashes, do the following:
```
conda activate rayjay_clone
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm jraph matplotlib tqdm optax
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flax==0.8.1 igraph unipath wandb==0.15.0
pip install tfp-nightly inference_gym
pip install numpyro
```

Additionally install Pytheus:
```
pip install pytheusQ
```

#### Brownian good config
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Brownian --n_integration_steps 100 --T_start 1. --T_end 1. --batch_size 1000 --lr 0.0002 --Energy_lr 0.0 --SDE_lr  0.0001 --N_anneal 6000 --feature_dim 64 --n_hidden 128 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

### Seeds good config
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Seeds --n_integration_steps 100 --T_start 1. --T_end 1. --batch_size 300 --lr 0.0003 --Energy_lr 0.0 --SDE_lr 0.0001 --N_anneal 6000 --feature_dim 64 --n_hidden 128 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

### Lorenz good config
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Lorenz --n_integration_steps 100 --T_start 1. --T_end 1. --batch_size 250 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 128 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0
```

### Ionosphere good config
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Ionosphere --n_integration_steps 100 --T_start 1. --T_end 1. --batch_size 1000 --lr 0.0003 --Energy_lr 0.0 --SDE_lr 0.0001 --N_anneal 6000 --feature_dim 64 --n_hidden 128 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```








python main.py --SDE_Loss LogVariance_Loss --Energy_Config Seeds --n_integration_steps 100 --T_start 4.1 --T_end 1. --batch_size 48 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 12000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 5. --use_interpol_gradient --Network_Type FeedForward --project_name iter --no-use_normal 