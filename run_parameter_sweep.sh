#!/bin/bash

# Define parameter ranges
learning_rates=(0.004 0.006 0.008 0.01)
sde_learning_rates=(0.001 0.002 0.003 0.004)
sigma_inits=(1.0 1.5 2.0 2.5)
beta_maxs=(0.05 0.1 0.15 0.2)

# Base configuration
base_config="--SDE_Loss Bridge_rKL_logderiv --Energy_Config LGCP --n_integration_steps 128 --T_start 1.0 --T_end 1. 
--batch_size 2000 --N_anneal 4000 --base_net PISNet --feature_dim 64 --n_hidden 64 --GPU 6 --beta_min 0.01 --use_interpol_gradient 
--Network_Type FeedForward --project_name grid_search --use_normal --SDE_Type Bridge_SDE"

# Create results directory
mkdir -p sweep_results

# Run experiments
for lr in "${learning_rates[@]}"; do
    for sde_lr in "${sde_learning_rates[@]}"; do
        for sigma_init in "${sigma_inits[@]}"; do
            for beta_max in "${beta_maxs[@]}"; do
                experiment_name="lr${lr}_sdelr${sde_lr}_sigma${sigma_init}_beta${beta_max}"
                echo "Running experiment: $experiment_name"
                
                python main.py $base_config \
                    --lr $lr \
                    --SDE_lr $sde_lr \
                    --sigma_init $sigma_init \
                    --beta_max $beta_max \
                    > "sweep_results/${experiment_name}.log" 2>&1
                
                echo "Completed experiment: $experiment_name"
            done
        done
    done
done

echo "Parameter sweep completed!" 