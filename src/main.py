import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from typing import List, Union
import torch
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from .experiments.experiment_ccsc import ExperimentCCSC

def main(data_name: str):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets: diabetes, haberman, hepatitis, hypothyroid, wdbc

    # Construct data file paths
    data_train_path = "".join(["src/data/csv/", data_name, "_train.csv"])
    data_test_path = "".join(["src/data/csv/", data_name, "_test.csv"])

    config_file_path = "".join(["src/config/model/", data_name, ".yaml"])
    # config_file_path = "src/config/model/model_toy.yaml"

    num_experiment = 100
    sparse_errs = torch.ones(num_experiment)
    cond_errs_wo = torch.ones(num_experiment)
    cond_errs = torch.ones(num_experiment)
    cond_svm_errs = torch.ones(num_experiment)
    coverages = torch.ones(num_experiment)
    header = "main -"

    # Load the data
    data_train = torch.tensor(
        pd.read_csv(data_train_path).to_numpy(), 
        dtype=torch.float32
    ).to(device)

    data_test = torch.tensor(
        pd.read_csv(data_test_path).to_numpy(), 
        dtype=torch.float32
    ).to(device)

    # for eid in tqdm(range(num_experiment),desc=" ".join([header, "running experiments"])):
    for eid in range(num_experiment):
        # Initialize the experiment
        experiment = ExperimentCCSC(
            prev_header=header + ">",
            experiment_id=eid, 
            config_file_path=config_file_path,
            device=device
        )

        # Run the experiment
        res = experiment(
            data_train,
            data_test
        )

        # Record the result error measures
        sparse_errs[eid] = res[0][1]
        cond_errs_wo[eid] = res[1][1]
        cond_errs[eid], coverages[eid] = res[2][1]
        cond_svm_errs[eid], _ = res[3][1]

        print(cond_errs)
        print(cond_svm_errs)
        # Print the results in a table format
        table = [
            ["Classifier Type", "Data", "Trials", "Min ER", "Min Cover", "Med ER", "Med Cover", "95th ER", "95th Cover", "Avg ER", "Avg Cover", "95th Avg ER", "95th Avg ER"],
            get_statistics("Classic Sparse", data_name, eid + 1, sparse_errs[:eid + 1]),
            get_statistics("Cond Sparse w/o Selector", data_name, eid + 1, cond_errs_wo[:eid + 1]),
            get_statistics("Cond Sparse", data_name, eid + 1, cond_errs[:eid + 1], coverages[:eid + 1]),
            get_statistics("Cond Sparse", data_name, eid + 1, cond_svm_errs[:eid + 1], coverages[:eid + 1])
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
    
def get_statistics(
        classifier: str,
        data_name:str,
        eid: int,
        errors: torch.Tensor,
        coverage: torch.Tensor = None
) -> List[Union[str, torch.Tensor]]:
    
    # if eid == 1:
    #     return [classifier, data_name, eid, errors, coverage, errors, coverage, errors, coverage, errors, coverage, errors, coverage]
    
    # min err
    min_err, min_ids = torch.min(errors, dim=0)
    
    # median err
    med_err, med_ids = torch.median(errors, dim=0)

    # sorting for computing 95th quantile statistics
    sorted_err, sorted_ids = torch.sort(errors)

    # 95th quantile err
    nfq_err = torch.quantile(sorted_err, q=0.95, interpolation='lower')
    print(sorted_err, nfq_err)

    # average err
    avg_err = torch.mean(errors)

    # 95th quatile average err
    nfq_err_ids = torch.where(sorted_err == nfq_err)[0]
    if nfq_err_ids.size(0) > 1:
        nfq_err_ids = nfq_err_ids[0]
    print(nfq_err_ids)
    nf_avg_err = torch.mean(sorted_err[:nfq_err_ids + 1])

    # compute coverages
    min_coverage = coverage
    med_coverage = coverage
    nfq_coverage = coverage
    avg_coverage = coverage
    nf_avg_coverage = coverage

    if coverage is not None:
        min_coverage = coverage[min_ids]
        med_coverage = coverage[med_ids]
        sorted_cov = coverage[sorted_ids]
        
        nfq_coverage = sorted_cov[nfq_err_ids]
        avg_coverage = torch.mean(coverage)
        nf_avg_coverage = torch.mean(sorted_cov[:nfq_err_ids + 1])

    return [classifier, data_name, eid, min_err, min_coverage, med_err, med_coverage, nfq_err, nfq_coverage, avg_err, avg_coverage, nf_avg_err, nf_avg_coverage]

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis project.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to use.')

    args = parser.parse_args()
    main(args.data_name)