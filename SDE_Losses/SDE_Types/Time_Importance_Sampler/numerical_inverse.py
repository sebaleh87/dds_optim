import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
# Define the functions a(t) and b(t)
def a(t):
    return np.sin(2 * np.pi * t) + 1.5  # Ensures positive values

def b(t):
    return np.cos(2 * np.pi * t) + 2    # Ensures positive values

class NumericalIntSampler():

    def __init__(self, a , b, eps = 10**-4, n_integration_steps = 50) -> None:
        self.a = a
        self.b = b
        self.n_intervals = n_integration_steps
        self.eps = eps
        self.Z = self.calculate_Z(self.n_intervals)

    def get_dt_values(self):
        dt_prev = 1/self.n_intervals
        t_values = self.get_t_values(self.n_intervals)
        pdf_values = self.unnormalized_pdf(t_values)/self.Z
        dt_values = pdf_values * dt_prev
        return t_values, dt_values

    def calculate_Z(self, num_samples):
        t_values = self.get_t_values(num_samples)

        # Compute the normalized PDF for comparison
        pdf_values = self.unnormalized_pdf(t_values)
        return integrate.trapz(pdf_values, t_values)

    def get_t_values(self, num_points):
        t_values = np.linspace(0, 1, num_points)
        return t_values

    def unnormalized_pdf(self, t):
        return np.array(self.a(t) / self.b(t))

    # Compute and normalize the CDF
    def compute_normalized_cdf(self, t_min=0.0, t_max=1.0, num_points=1000):
        t_values = self.get_t_values(num_points)
        pdf_values = self.unnormalized_pdf(t_values)
        cdf_values = integrate.cumtrapz(pdf_values, t_values, initial=0)
        cdf_normalized = cdf_values / cdf_values[-1]
        return t_values, cdf_normalized

    # Create the inverse CDF function using interpolation
    def create_inverse_cdf(self, t_values, cdf_normalized):
        inverse_cdf_func = interpolate.interp1d(
            cdf_normalized, t_values,
            kind='linear',
            bounds_error=False,
            fill_value=(t_values[0], t_values[-1])
        )
        return inverse_cdf_func

    # Sampling via inverse CDF
    def sample_via_inverse_cdf(self, inverse_cdf_func, num_samples):
        uniform_samples = np.random.uniform(self.eps, 1, size=num_samples)
        samples = inverse_cdf_func(uniform_samples)
        return samples

    # Complete sampling function
    def numerical_inversion_sampling(self, num_samples=10000, num_points=1000):
        # Compute the normalized CDF
        t_values, cdf_normalized = self.compute_normalized_cdf(num_points=num_points, t_min = self.eps)
        
        # Create the inverse CDF function
        inverse_cdf_func = self.create_inverse_cdf(t_values, cdf_normalized)
        
        # Sample using the inverse CDF
        samples = self.sample_via_inverse_cdf(inverse_cdf_func, num_samples)
        
        return t_values, cdf_normalized, samples, inverse_cdf_func

    
    def visualize(self):

        num_samples = 10000
        t_values, cdf_normalized, samples, inverse_cdf_func = self.numerical_inversion_sampling(num_samples=num_samples)

        # Compute the normalized PDF for comparison
        pdf_values = self.unnormalized_pdf(t_values)
        pdf_normalized = pdf_values / integrate.trapz(pdf_values, t_values)

        # Visualization
        plt.figure(figsize=(12, 8))

        # Plot the normalized PDF
        plt.subplot(2, 1, 1)
        plt.plot(t_values, pdf_normalized, label='Normalized PDF', color='blue')
        plt.title('Unnormalized and Normalized PDF vs Sampled Histogram')
        plt.yscale('log')
        plt.xlabel('t')
        plt.ylabel('Probability Density')
        plt.legend()

        # Plot the histogram of the samples
        plt.subplot(2, 1, 2)
        plt.hist(samples, bins=50, density=True, alpha=0.6, color='orange', label='Sampled Histogram')
        plt.plot(t_values, pdf_normalized, 'r-', label='Normalized PDF')
        plt.yscale('log')
        plt.xlabel('t')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.savefig("numerical_inverse_sampling.png")

        plt.figure(figsize=(10, 6))

        # Plot the unnormalized distribution
        plt.subplot(3, 1, 1)
        plt.plot(t_values, pdf_normalized, label=r'$\frac{a(t)}{b(t)}$ (unnormalized)', color='blue')
        plt.title("Unnormalized Distribution vs Sampled Histogram")
        plt.yscale('log')
        plt.xlabel('t')
        plt.ylabel('Unnormalized Probability Density')
        plt.legend()
    
        t_values = np.linspace(0, 1, 1000)
        unnormalized_pdf =  self.b(t_values)


        # Plot the unnormalized distribution
        plt.subplot(3, 1, 2)
        plt.plot(t_values, unnormalized_pdf, label=r'$g(t)^2$ (unnormalized)', color='blue')
        plt.title("b")
        plt.xlabel('t')
        plt.ylabel('Unnormalized Probability Density')
        plt.legend()
    
        a = self.a(t_values) 

        # Plot the unnormalized distribution
        plt.subplot(3, 1, 3)
        plt.plot(t_values, a, label=r'$a(t)^2$ (unnormalized)', color='blue')
        plt.title("a")
        plt.xlabel('t')
        plt.ylabel('Unnormalized Probability Density')
        plt.legend()
        plt.savefig("before_sampling.png")

