import os
import pandas as pd
import requests
import numpy as np
import jax
from jax import numpy as jnp

# Function to download the dataset
def download_sonar_dataset():
    # URL of the dataset
    current_folder = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(current_folder, "sonar")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # File path to save the dataset
    file_path = os.path.join(save_folder, "sonar.csv")
    
    # Download and save the dataset
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Dataset downloaded and saved to {file_path}")
    else:
        raise Exception(f"Failed to download dataset. HTTP Status Code: {response.status_code}")
    
    return file_path

# Function to load the dataset into a Pandas DataFrame
def load_sonar_dataset(file_path):
    # Load the dataset
    column_names = [f"Feature_{i+1}" for i in range(60)] + ["Label"]
    data = pd.read_csv(file_path, header=None, names=column_names)
    print(f"Dataset loaded with shape: {data.shape}")
    return data


def compute_energy(data, labels, parameters):

    log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(parameters, loc=0, scale=1), axis = -1)
    log_bernoulli = jnp.sum(jax.scipy.stats.bernoulli.logpmf(labels, jax.nn.sigmoid(jnp.dot(data, parameters))), axis = -1)

    energy = -log_prior - log_bernoulli
    return energy

# Main execution
if __name__ == "__main__":
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{0}"
    # Define the folder to save the dataset
    folder = "datasets"
    
    # Download the dataset
    dataset_path = download_sonar_dataset()
    
    # Load the dataset
    sonar_data = load_sonar_dataset(dataset_path)
    
    # Display the first few rows
    print(sonar_data.head())

    # Convert the Pandas DataFrame to a NumPy array
    sonar_numpy_data = sonar_data.to_numpy()
    
    # Display the shape of the NumPy array
    print(f"NumPy array shape: {sonar_numpy_data.shape}")
    print(np.var(sonar_numpy_data[:, 0:-1], axis = 0))
    print(np.mean(sonar_numpy_data[:, 0:-1], axis = 0))

    data = np.array(sonar_numpy_data[:, 0:-1], dtype=np.float32)
    str_labels = sonar_numpy_data[:, -1]
    labels = np.array(jnp.where(str_labels == "R", 0, 1), dtype=np.float32)


    parameters = jnp.zeros(data.shape[1])

    compute_energy(data, labels, parameters)