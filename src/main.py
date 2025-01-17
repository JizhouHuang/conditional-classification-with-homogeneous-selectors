import torch
from tqdm import tqdm
import yaml
from tabulate import tabulate
from .experiments.experiment_ccsc import ExperimentCCSC
from .utils.helpers import UCIMedicalDataset

def main():

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the configuration from the YAML file
    with open('src/config/hepatitis.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration values
    data_file_path = config['data_file_path']
    config_file_path = config['config_file_path']
    attribute_names = config['attribute_names']
    categorical_attr_names = config['categorical_attr_names']
    label_true = config['label_true']
    label_false = config['label_false']

    num_experiment = 2
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
            config_file_path=config_file_path
        ).to(device)

        # Load and preprocess the data
        uci_data = UCIMedicalDataset(
            file_path=data_file_path,
            attributes=attribute_names,
            categorical_attr=categorical_attr_names,
            label_true=label_true,
            label_false=label_false,
            device=device
        )
        data_train, data_test = uci_data.SliceDataWithRatio(0.8)

        # Run the experiment
        res = experiment(data_train, data_test)
        sparse_errs[experiment_id] = res[0][1]
        cond_errs_wo[experiment_id] = res[1][1]
        cond_errs[experiment_id], coverages[experiment_id] = res[2][1]
    
    # Print the results in a table format
    table = [
        ["Classifier Type", "Number of Experiments", "Average Est Error Rate", "Average Coverage"],
        ["Classic Sparse", num_experiment, torch.mean(sparse_errs), 1],
        ["Conditional Sparse w/o Selector", num_experiment, torch.mean(cond_errs_wo), 1],
        ["Conditional Sparse", num_experiment, torch.mean(cond_errs), torch.mean(coverages)]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

if __name__ == "__main__":
    main()