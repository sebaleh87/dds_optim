import jax.numpy as jnp

class MovAvrgCalculator:
    def __init__(self, alpha):
        self.alpha = alpha
        self.mean_moving_avg = None
        self.std_moving_avg = None

    def compute_averages(self, Energy_values):
        mean = jnp.mean(Energy_values)
        std = jnp.std(Energy_values)

        if self.mean_moving_avg is None:
            self.mean_moving_avg = mean
        else:
            self.mean_moving_avg = self.alpha * mean + (1 - self.alpha) * self.mean_moving_avg

        if self.std_moving_avg is None:
            self.std_moving_avg = std
        else:
            self.std_moving_avg = self.alpha * std + (1 - self.alpha) * self.std_moving_avg

        return mean, std, self.mean_moving_avg, self.std_moving_avg
    

if(__name__ == "__main__"):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    alphas = [0.001, 0.05, 0.01, 0.005]
    for alpha in alphas:
        mov_avg = MovAvrgCalculator(alpha)

        # Generate sinusoidal data with noise
        x = np.linspace(0, 100, 10000)
        noise = np.random.normal(0, 0.1, x.shape)
        y = 20*np.sin(x) - x + noise

        means = []
        stds = []
        mean_moving_avgs = []
        std_moving_avgs = []

        for value in y:
            mean, std, mean_moving_avg, std_moving_avg = mov_avg.compute_averages(np.array([value]))
            means.append(mean)
            stds.append(std)
            mean_moving_avgs.append(mean_moving_avg)
            std_moving_avgs.append(std_moving_avg)

        plt.figure(figsize=(14, 7))
        plt.plot(x, y, label='Noisy Sinusoidal Data')
        plt.plot(x, mean_moving_avgs, label='Mean Moving Average', linestyle='--')
        plt.plot(x, std_moving_avgs, label='Std Moving Average', linestyle='--')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Moving Averages of Noisy Sinusoidal Data')

        plt.savefig(os.getcwd()  + f"/Figures/mov_avrg_{alpha}.png")
