#Installation

```
conda env create -f environment.yml
```

When installation crashes, do the following:
```
conda env create --file=environment.yml 
conda activate dds_optim_new
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


