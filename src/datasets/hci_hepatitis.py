import pandas as pd
import torch

# Load the dataset
file_path = "hepatitis.data"  # Replace with your file path
column_names = [
    "Class",  # Target (DIE=1, LIVE=2)
    "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
    "Anorexia", "Liver Big", "Liver Firm", "Spleen Palpable", "Spiders",
    "Ascites", "Varices", "Bilirubin", "Alk Phosphate", "SGOT",
    "Albumin", "Protime", "Histology"
]

# Load data, replacing missing values ("?") with NaN
data = pd.read_csv(file_path, header=None, names=column_names, na_values="?")

# Display basic info
print(data.head())
print(data.isnull().sum())  # Check for missing values
