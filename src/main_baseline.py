import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from .experiments.experiment_baseline import ExperimentBaseline
from .utils.data import UCIMedicalDataset
from .models.baseline_learner import LogisticRegLearner, SVMLearner, RandomForestLearner, XGBoostLearner

def main(data_name: str):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets: diabetes, haberman, hepatitis, hypothyroid, wdbc
    # Construct data file paths
    data_train_path = "".join(["src/data/", data_name, "_train.pkl"])
    data_test_path = "".join(["src/data/", data_name, "_test.pkl"])
    config_file_path = "".join(["src/config/model/", data_name, ".yaml"])
    # config_file_path = "src/config/model/model_toy.yaml"

    num_experiment = 100
    learners = [LogisticRegLearner, SVMLearner, RandomForestLearner, XGBoostLearner]
    errs = torch.ones([num_experiment, len(learners)])
    cond_errs = torch.ones([num_experiment, len(learners)])
    coverages = torch.ones([num_experiment, len(learners)])
    header = "main -"

    for experiment_id in tqdm(range(num_experiment),desc=" ".join([header, "running experiments"])):
        # Initialize the experiment
        experiment = ExperimentBaseline(
            prev_header=header + ">",
            experiment_id=experiment_id, 
            config_file_path=config_file_path,
            device=device
        )

        # Load the data
        data_train = torch.tensor(
            pd.read_pickle(data_train_path).to_numpy(), 
            dtype=torch.float32
        ).to(device)

        data_test = torch.tensor(
            pd.read_pickle(data_test_path).to_numpy(), 
            dtype=torch.float32
        ).to(device)

        # Run the experiment
        # errs[experiment_id], cond_errs[experiment_id], coverages[experiment_id] = experiment(
        #     data_train, 
        #     data_test, 
        #     learners
        # )

        errs[experiment_id], cond_errs[experiment_id], coverages[experiment_id] = experiment(
            data_train[:min(10000, data_train.size(0))],
            data_test[:min(10000, data_test.size(0))],
            learners
        )
    
    min_errs, _ = torch.min(errs, dim=0)
    avg_errs = torch.mean(errs, dim=0)
    min_cond_errs, min_cond_ids = torch.min(cond_errs, dim=0)
    min_coverages = coverages[min_cond_ids, torch.arange(len(learners))]
    avg_cond_errs = torch.mean(cond_errs, dim=0)
    avg_coverages = torch.mean(coverages, dim=0)
    # Print the results in a table format

    table = [
        ["Predictor Name", "Data", "Trials", "Min ER", "Avg ER", "Min CER", "Min Coverage", "Avg CER", "Avg Coverage"],
        ["Logistic", data_name, num_experiment, min_errs[0], avg_errs[0], min_cond_errs[0], min_coverages[0], avg_cond_errs[0], avg_coverages[0]],
        ["SVM", data_name, num_experiment, min_errs[1], avg_errs[1], min_cond_errs[1], min_coverages[1], avg_cond_errs[1], avg_coverages[1]],
        ["Random ForestRandom Forest", data_name, num_experiment, min_errs[2], avg_errs[2], min_cond_errs[2], min_coverages[2], avg_cond_errs[2], avg_coverages[2]],
        ["XGBoost", data_name, num_experiment, min_errs[3], avg_errs[3], min_cond_errs[3], min_coverages[3], avg_cond_errs[3], avg_coverages[3]]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis project.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to use.')

    args = parser.parse_args()
    main(args.data_name)