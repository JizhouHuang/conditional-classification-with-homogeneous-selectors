import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from tqdm import tqdm
import yaml
from tabulate import tabulate
from .experiments.experiment_baseline import ExperimentBaseline
from .utils.data import UCIMedicalDataset
from .models.baseline_learner import LogisticRegLearner, SVMLearner, RandomForestLearner, XGBoostLearner

def main(data_name: str):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the configuration from the YAML file
    # Datasets: diabetes, haberman, hepatitis, hypothyroid, wdbc
    experiment_config_file_path = "".join(["src/config/data/", data_name, ".yaml"])
    with open(experiment_config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration values
    data_file_path = config['data_file_path']
    config_file_path = config['config_file_path']
    attribute_names = config['attribute_names']
    label_name = config['label_name']
    categorical_attr_names = config['categorical_attr_names']
    binary_attr_names = config['binary_attr_names']
    sparse_attr_names = config['sparse_attr_names']
    label_true = config['label_true']
    label_false = config['label_false']
    attr_true = config['attr_true']
    attr_false = config['attr_false']

    num_experiment = 1
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

        # Load and preprocess the data
        uci_data = UCIMedicalDataset(
            file_path=data_file_path,
            attributes=attribute_names,
            label_name=label_name,
            categorical_attr_names=categorical_attr_names,
            binary_attr_names=binary_attr_names,
            sparse_attr_names=sparse_attr_names,
            label_true=label_true,
            label_false=label_false,
            attr_true=attr_true,
            attr_false=attr_false,
            device=device
        )

        # Run the experiment
        errs[experiment_id], cond_errs[experiment_id], coverages[experiment_id] = experiment(uci_data, learners)
    
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