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
```

Additionally install Pytheus:
```
pip install pytheusQ
```

#Start experiment on Rasragin Problem
```
python main.py --SDE_Loss Discrete_Time_rKL_Loss --Energy_Config Rastrigin --n_integration_steps 10 --T_start 15 --batch_size 200 --lr 0.002
```

# Experiment on Pytheus
```
python main.py main.py --SDE_Loss Discrete_Time_rKL_Loss --Energy_Config Pytheus --n_integration_steps 10 --T_start 0.5 --batch_size 400 --lr 0.001 --N_anneal 1000 --feature_dim 0 --n_hidden 200
```