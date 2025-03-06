#!/bin/bash

(
    # Step 1: Initialize the sweep and retrieve the sweep ID
    SWEEP_OUTPUT=$(wandb sweep ./Configs/Sweeps/GMM/Anneal.yaml 2>&1) # Capture stdout and stderr
    AGENT_COMMAND=$(echo "$SWEEP_OUTPUT" | grep -oP 'Run sweep agent with: \K.*')
    echo "Command to start agents: $AGENT_COMMAND" 

    # Check if the SWEEP_ID is valid
    if [ -z "AGENT_COMMAND" ]; then
        echo "Error: AGENT_COMMAND could not be retrieved. Please check the output below:"
        echo "$SWEEP_OUTPUT"
        exit 1
    fi

    # Step 2: Define the number of agents per GPU
    AGENTS_PER_GPU=1
    SESSION_PREFIX="sweep_"

    # Function to start wandb agents on a specific GPU in a tmux session
    start_agents() {
        GPU_ID=$1
        NUM_AGENTS=$2

        echo "Starting $NUM_AGENTS agents on GPU $GPU_ID..."

        for (( i=0; i<=NUM_AGENTS-1; i++ ))
        do
            SESSION_NAME="${SESSION_PREFIX}${GPU_ID}${i}"

            # Start a new tmux session and run the wandb agent in it
            tmux new-session -d -s "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=$GPU_ID $AGENT_COMMAND"
            
            echo "Started agent in tmux session: $SESSION_NAME on GPU $GPU_ID"
        done
    }

    # Step 3: Start agents on GPU 0
    #start_agents 0 $AGENTS_PER_GPU
    start_agents 1 $AGENTS_PER_GPU
    # start_agents 2 $AGENTS_PER_GPU
    # start_agents 3 $AGENTS_PER_GPU
    # start_agents 4 $AGENTS_PER_GPU
    start_agents 5 $AGENTS_PER_GPU
    #start_agents 6 $AGENTS_PER_GPU
    #start_agents 7 $AGENTS_PER_GPU

    echo "All agents started in separate tmux sessions."

)
# use "tmux kill-server" to kill sweep agents