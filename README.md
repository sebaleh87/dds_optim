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

## List of Hyperparameters that should be tuned:
#### Learning rate of Network Hyperparameters:
```
--lr 0.0005
```

#### Learning rate of SDE Hyperparameters:
```
 --SDE_lr 0.0005
```

#### maximum and Minimum value of diffusion constant schedule:
For VP-SDE (```--SDE_Type VP_SDE```) it should be something like:
```
 --beta_max 5. --beta_min 0.05
```

and for VE-SDE (```--SDE_Type VE_SDE```) it should be something like:
```
 --beta_max 1. --beta_min 0.01
```

These hyperparameters probably do not need to be tuned but you should use the correct default values depending on the SDE.

#### Scale of the diffusion parameter
```
 --sigma_init 1.
```

## Sonar good config

(The larger the ELBO the better)

DDS is similar to our VP-SDE and PIS is similar to our VE-SDE

ELBO Literature DDS: -121.22 \
ELBO Literature PIS: -142 \
Best known ELBO Literature: -108.18 \
(from SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS) \

#### VP-SDE
ELBO of this config: -111.5 \
```
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Sonar --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 5. --beta_min 0.05 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VP_SDE --repulsion_strength 0.0
```

#### Bridge
ELBO of this config: -109.4 \
```
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Sonar --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.05 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0
```


## Brownian good config 

ELBO Literature DDS: 0.56 \
ELBO Literature PIS: N/A \
Best known ELBO Literature: 1.00 \
(from SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS) \

#### VE-SDE
ELBO of this config: -0.94 \
```
 python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0002 --Energy_lr 0.0 --SDE_lr  0.0002 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```


## Seeds good config
ELBO Literature DDS: -75.21 \
ELBO Literature PIS: -88.92 \
Best known ELBO Literature: -73.45 \
(from SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS) \

#### VE-SDE
ELBO of this config: -74.5 \
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

## Lorenz good config

ELBO Literature DDS: Unknown \
ELBO Literature PIS: Unknown \
Best known ELBO Literature: 1153.1 \
(from Langevin Diffusion Variational Inference) \

#### VE-SDE
ELBO of this config: 1336 \
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Lorenz --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0
```

## Ionosphere good config
ELBO Literature DDS: Unknown \
ELBO Literature PIS: Unknown \
Best known ELBO Literature: −111.9 \
(from Langevin Diffusion Variational Inference) \

#### VE-SDE 
ELBO of this config: -113.0 \
```
main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Ionosphere --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

