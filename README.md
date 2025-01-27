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
pip install scikit-learn
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
For VP-SDE (```--SDE_Type VP_SDE```) it should be like:
```
 --beta_max 5. --beta_min 0.05
```

and for VE-SDE (```--SDE_Type VE_SDE```) it should be like:
```
 --beta_max 1. --beta_min 0.01
```

and for Bridge-SDE (```--SDE_Type Bridge_SDE```) it should be like:
```
 --beta_max 0.1 --beta_min 0.01
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
(from SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS) 

#### VP-SDE
ELBO of this config: -111.5 \
```
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Sonar --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 5. --beta_min 0.05 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VP_SDE --repulsion_strength 0.0
```

#### Bridge

Bridge shorter training as in SLCD: ELBO = -108.83
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Sonar --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --model_seed 0 1 2 
```


## Brownian good config 

ELBO Literature DDS: 0.56 \
ELBO Literature PIS: N/A \
Best known ELBO Literature: 1.00 \
(from SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS) 

#### VE-SDE
ELBO of this config: -0.96 \
```
 python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0002 --Energy_lr 0.0 --SDE_lr  0.0002 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

rKL w/ logderiv
```
 python main.py --SDE_Loss Reverse_KL_Loss_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0002 --Energy_lr 0.0 --SDE_lr  0.0002 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```


### Bridge
Bridge rKL LD learned params: ELBO = 1.00
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name stability_test --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
```

Bridge rKL LogVar learned params: divergend
```
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name stability_test --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
```

Bridge rKL LD fixed params: 
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --SDE_lr 0.001 --learn_SDE_params_mode prior_only  --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name stability_test --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
```

Bridge rKL LogVar fixed params: 
```
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --learn_SDE_params_mode prior_only --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name stability_test --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
```


## Seeds good config
ELBO Literature DDS: -75.21 \
ELBO Literature PIS: -88.92 \
Best known ELBO Literature: -73.45 \
(from SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS) 

#### VE-SDE 
ELBO of this config: -74.5 \
```
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

rKL w/ logderiv

```
python main.py --SDE_Loss Reverse_KL_Loss_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

### Bridge
Bridge shorter training as in SLCD: ELBO = -73.48

```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Seeds --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name paper_runs --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.3 --model_seed 0 1 2
```

## Lorenz good config

ELBO Literature DDS: Unknown \
ELBO Literature PIS: Unknown \
Best known ELBO Literature: 1153.1 \
(from Langevin Diffusion Variational Inference) 

#### VE-SDE
ELBO of this config: 1336 \
```
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Lorenz --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0
```

## Ionosphere good config
ELBO Literature DDS: Unknown \
ELBO Literature PIS: Unknown \
Best known ELBO Literature: −111.9 \
(from Langevin Diffusion Variational Inference) 

#### VE-SDE 
ELBO of this config: -113.0 \
```
python main.py --SDE_Loss Reverse_KL_Loss --Energy_Config Ionosphere --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE --repulsion_strength 0.0 --sigma_init 0.1
```

#### VE-SDE rKL w/ logderiv
ELBO of this config: 
```
python main.py --SDE_Loss Reverse_KL_Loss_logderiv --Energy_Config Ionosphere --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 1. --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type VE_SDE  --sigma_init 0.1
```

### Bridge
Bridge rKL LD:

```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Ionosphere --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.002 --Energy_lr 0.0 --SDE_lr 0.002 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name stability_test --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.
```

Bridge rKL LV:

```
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Ionosphere --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.002 --Energy_lr 0.0 --SDE_lr 0.002 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name stability_test --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.
```



## GaussianMixture 2-D

### Brige CMCD (Not our method!)
```
python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1. --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.0 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.01 --beta_min 0.0 --use_interpol_gradient --Network_Type FeedForward --project_name log_deriv_rKL_off_policy --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 20.
```

#### Bridge
ELBO of this config: - 0.5
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 8. --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name log_deriv --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1.
```

with noise 
ELBO = -0.3

```--sigma_scale_factor 1.``` should be 1 and sigma scale strength is specified with ```--T_start``` where 1- T_start specifies the scale strength

```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1.01 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name log_deriv_rKL_off_policy --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --use_off_policy --sigma_scale_factor 1.
```


## Gaussian Mixture 50-D
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GaussianMixture --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.01 --SDE_lr 0.0002 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 2.0 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name final_runs --use_normal --SDE_Type Bridge_SDE --sigma_init 40. --n_particles 50 --clip_value 1.
```


## Funnel good config

ELBO Literature DDS: -0.597 \
ELBO Literature PIS: -3.198 \
Best known ELBO Literature: -0.011

#### VE-SDE

### VP-SDE

### Bridge
Elbo of this config: -0.245 \
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Funnel --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --n_particles 10 ```

## MW54 good config

ELBO Literature DDS: Unknown \
ELBO Literature PIS: Unknown \
Best known ELBO Literature: Unknown


#### VE-SDE

### VP-SDE

### Bridge
ELBO of this config: 12.14 \
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config MW54 --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1. --n_particles 10
```

## LGCP good config

ELBO Literature DDS: Unknown \
ELBO Literature PIS: 479.54 \
Best known ELBO Literature: 497.85

#### VE-SDE

### VP-SDE

### Bridge
ELBO of this config: 461.25 \
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 1000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 2 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 2.```


## German Credit good config

ELBO Literature DDS: -514.74 \
ELBO Literature PIS: -846.74 \
Best known ELBO Literature: -504.46

#### VE-SDE

### VP-SDE

### Bridge
ELBO of this config: -504.62 \
```
python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config GermanCredit --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.001 --Energy_lr 0.0 --SDE_lr 0.001 --N_anneal 6000 --feature_dim 64 --n_hidden 64 --GPU 6 --beta_max 0.1 --beta_min 0.01 --use_interpol_gradient --Network_Type FeedForward --project_name FeedForward --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 1
```