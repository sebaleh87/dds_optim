#Installation


```
conda env create -f environment.yml
conda activate rayjay_clone
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm jraph matplotlib tqdm optax
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flax==0.8.1 igraph unipath wandb==0.15.0
pip install pytheusQ
```


#Start experiment on Rasragin Problem
```
python main.py --SDE_Loss Discrete_Time_rKL_Loss --Energy_Config Rastrigin --n_integration_steps 10 --T_start 15 --batch_size 200 --lr 0.002
```