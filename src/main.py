import pandas as pd
import numpy as np

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values."""
    return df.dropna()

def summarize_data(df):
    """Generate summary statistics of the dataset."""
    return df.describe()

def main():
    # Load the dataset
    data = load_data('data.csv')
    
    # Clean the dataset
    clean_data_df = clean_data(data)
    
    # Summarize the dataset
    summary = summarize_data(clean_data_df)
    
    # Output the results
    print(summary)

if __name__ == "__main__":
    main()