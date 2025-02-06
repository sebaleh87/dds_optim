import wandb
import numpy as np
import time


def test_wandb_logging():
    try:
        # Initialize wandb
        wandb.init(
            project="sampling",
            entity="bartmann-jku-linz",
            settings=wandb.Settings(base_url="https://api.wandb.ai")
        )
        
        print("Successfully initialized wandb")
        
        # Try to log some simple metrics
        for i in range(10):
            metrics = {
                "test_metric": np.random.rand(),
                "iteration": i,
                "sine_wave": np.sin(i/10.0)
            }
            wandb.log(metrics)
            print(f"Logged metrics for step {i}: {metrics}")
            time.sleep(1)  # Wait a second between logs
            
        print("Successfully logged all metrics")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        wandb.finish()

if __name__ == "__main__":

    # print("Starting wandb test...")
    # print(f"Wandb version: {wandb.__version__}")
    # test_wandb_logging()

    import jax
    print(jax.devices())  # Should show GPU devices if properly installed