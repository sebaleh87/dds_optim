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
```

Additionally install Pytheus:
```
pip install pytheusQ
```

For Seeds dataset support, install additional packages:
```
pip install numpyro==0.16.1
pip install --upgrade jax==0.4.37
pip install tensorflow-probability[tf]==0.25.0
pip install inference-gym==0.0.4
```

#Start experiment on Rasragin Problem
```
python main.py --SDE_Loss Discrete_Time_rKL_Loss --Energy_Config Rastrigin --n_integration_steps 10 --T_start 15 --batch_size 200 --lr 0.002
```

# Experiment on Pytheus
```
python main.py main.py --SDE_Loss Discrete_Time_rKL_Loss --Energy_Config Pytheus --n_integration_steps 10 --T_start 0.5 --batch_size 400 --lr 0.001 --N_anneal 1000 --feature_dim 0 --n_hidden 200
```

	
3:16 PM








python main.py --SDE_Loss LogVariance_Loss --Energy_Config Seeds --n_integration_steps 100 --T_start 4.1 --T_end 1. --batch_size 48 --lr 0.0005 --Energy_lr 0.0 --SDE_lr 0.0005 --N_anneal 12000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 5. --use_interpol_gradient --Network_Type FeedForward --project_name iter --no-use_normal 