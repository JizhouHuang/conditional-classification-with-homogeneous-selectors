import pandas as pd
import torch

def load_data(file_path, k=None):
    df = pd.read_csv(file_path)
    if k is not None and k < len(df):
        df = df.sample(n=k)
    return df

def clean_data(df):
    df.dropna(inplace=True)
    return df

def summarize_data(df):
    return df.describe()

def fake_data(num_sample, num_feature):
    real_values = torch.randn(num_sample, num_feature)  # Standard normal distribution
    boolean_values = torch.randint(0, 2, (num_sample, 1), dtype=torch.bool)
    return torch.cat((real_values, boolean_values.float()), dim=1)