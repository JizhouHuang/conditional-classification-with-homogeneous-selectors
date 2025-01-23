import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from .experiments.experiment_ccsc import ExperimentCCSC
from .utils.data import UCIMedicalDataset

def main(data_name: str):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets: diabetes, haberman, hepatitis, hypothyroid, wdbc

    # Construct data file paths
    data_train_path = "".join(["src/data/csv/", data_name, "_train.csv"])
    data_test_path = "".join(["src/data/csv/", data_name, "_test.csv"])
    # data_train_path = "".join(["src/data/", data_name, "_train.pkl"])
    # data_test_path = "".join(["src/data/", data_name, "_test.pkl"])
    # config_file_path = "".join(["src/config/model/", data_name, ".yaml"])
    config_file_path = "src/config/model/model_toy.yaml"

    num_experiment = 1
    sparse_errs = torch.ones(num_experiment)
    cond_errs_wo = torch.ones(num_experiment)
    cond_errs = torch.ones(num_experiment)
    coverages = torch.ones(num_experiment)
    header = "main -"

    for experiment_id in tqdm(range(num_experiment),desc=" ".join([header, "running experiments"])):
        # Initialize the experiment
        experiment = ExperimentCCSC(
            prev_header=header + ">",
            experiment_id=experiment_id, 
            config_file_path=config_file_path,
            device=device
        )

        # Load the data
        data_train = torch.tensor(
            pd.read_csv(data_train_path).to_numpy(), 
            dtype=torch.float32
        ).to(device)

        data_test = torch.tensor(
            pd.read_csv(data_test_path).to_numpy(), 
            dtype=torch.float32
        ).to(device)

        # Run the experiment
        res = experiment(
            data_train[:min(1000, data_train.size(0))],
            data_test[:min(1000, data_test.size(0))]
        )
        # res = experiment(
        #     data_train,
        #     data_test
        # )

        # Record the result error measures
        sparse_errs[experiment_id] = res[0][1]
        cond_errs_wo[experiment_id] = res[1][1]
        cond_errs[experiment_id], coverages[experiment_id] = res[2][1]
    
    min_cond_err, min_cond_ind = torch.min(cond_errs, dim=0)
    # Print the results in a table format
    table = [
        ["Classifier Type", "Data", "Trials", "Min Est ER", "Min Coverage", "Avg Est ER", "Avg Coverage"],
        ["Classic Sparse", data_name, num_experiment, torch.min(sparse_errs), 1, torch.mean(sparse_errs), 1],
        ["Cond Sparse w/o Selector", data_name, num_experiment, cond_errs_wo[min_cond_ind], 1, torch.mean(cond_errs_wo), 1],
        ["Cond Sparse", data_name, num_experiment, min_cond_err, coverages[min_cond_ind], torch.mean(cond_errs), torch.mean(coverages)]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis project.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to use.')

    args = parser.parse_args()
    main(args.data_name)