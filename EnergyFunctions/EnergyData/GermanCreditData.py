import os
import pandas as pd
import requests
import numpy as np

# Function to download the dataset
def download_german_credit_dataset(save_folder="datasets"):
    # URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # File path to save the dataset
    file_path = os.path.join(save_folder, "german_credit.csv")
    
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
def load_german_credit_dataset(file_path):
    # Define column names based on the dataset description
    column_names = [
        "Status", "Duration", "Credit_history", "Purpose", "Credit_amount",
        "Savings", "Employment", "Installment_rate", "Personal_status", "Debtors",
        "Residence", "Property", "Age", "Other_installment_plans", "Housing",
        "Existing_credits", "Job", "Liable_people", "Telephone", "Foreign_worker", "Label"
    ]
    
    # Load the dataset
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)
    print(f"Dataset loaded with shape: {data.shape}")
    return data

# Main execution
if __name__ == "__main__":
    # Define the folder to save the dataset
    folder = "datasets"
    
    # Download the dataset
    dataset_path = download_german_credit_dataset(save_folder=folder)
    
    # Load the dataset
    german_credit_data = load_german_credit_dataset(dataset_path)
    
    # Display the first few rows
    print(german_credit_data.head())
    numpy_credit = np.array(german_credit_data)
    print(numpy_credit.shape)
