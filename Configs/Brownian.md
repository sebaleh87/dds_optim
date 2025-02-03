  DDS_Brownian_stability_final_dim_32
  
    ''' LogVarLoss all SDE params: wandb_id divergent
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5
    '''

    ''' LogVarLoss only prior: wandb_id 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5 --learn_SDE_params_mode prior_only
    '''

    ''' Bridge_rKL_logderiv only prior: wandb_id 
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5 --learn_SDE_params_mode prior_only
    '''

    rKL_all_SDE_ids_higher_sigma = ["leafy-silence-41", "logical-totem-34", "noble-salad-23"]
    ''' Bridge_rKL_logderiv learn SDE params: wandb_id "noble-salad-23"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5
    '''


    #### Convergent parameters
    LV_all_SDE_ids = ["zany-vortex-45", "wobbly-serenity-40", "vocal-feather-27"]
    ''' LogVarLoss all SDE params: wandb_id = 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Interpol_lr 0.0001 --SDE_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 7 --beta_max 0.025 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 
    '''

    LV_only_prior = ["tough-eon-43", "cosmic-plasma-38", "devout-aardvark-31"]
    ''' LogVarLoss only prior: wandb_id = 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Interpol_lr 0.0001 --SDE_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.025 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 --learn_SDE_params_mode prior_only
    '''

    rKL_only_prior = ["driven-grass-42", "fearless-smoke-37", "astral-river-20"]
    ''' Bridge_rKL_logderiv only prior: wandb_id = "astral-river-20"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 --learn_SDE_params_mode prior_only
    '''

    rKL_all_SDE_ids = ["tough-sky-44","twilight-elevator-39", "lyric-fog-19"]
    ''' Bridge_rKL_logderiv learn SDE params: wandb_id = "lyric-fog-19"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
    '''



### init at mean

python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.02 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5 --model_seeds 0 1 2 --learn_SDE_params_mode prior_only