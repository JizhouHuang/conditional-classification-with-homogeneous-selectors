import torch
import torch.nn as nn
import yaml
from typing import Callable, Union, Tuple
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from ..utils.helpers import Classify, SparseClassify, TransformedDataset
from ..models.conditional_classification_finite_class import ConditionalLearnerForFiniteClass
from ..models.robust_list_learning_of_sparse_linear_classifiers import RobustListLearner

class Experiment(nn.Module):
    def __init__(
            self,
            experiment_id: int,
            config_file_path:str
    ):
        super(Experiment, self).__init__()
        self.header = " ".join(["experiment", str(experiment_id), "-"])

        # read yaml file from config_file_path   
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.data_frac_rll = config['data_frac_rll']
        self.margin = config['margin']
        self.sparsity = config['sparsity']
        self.num_cluster = config['num_cluster']
        self.data_frac_psgd = config['data_frac_psgd']
        self.lr_coeff = config['lr_coeff']
        self.num_iter = config['num_iter']
        self.batch_size = config['batch_size']

    def forward(
            self,
            data_train: torch.Tensor,
            data_test: torch.Tensor
    ) -> None:
        # Learn the sparse classifiers
        print(" ".join([self.header, "initializing robust list learner for sparse perceptrons ..."]))
        robust_list_learner = RobustListLearner(
            prev_header=self.header + ">",
            sparsity=self.sparsity, 
            margin=self.margin,
            num_cluster=self.num_cluster
        ).to(data_train.device)

        list_learning_sample_size = int(self.data_frac_rll * data_train.shape[0])
        rl_dataloader = DataLoader(
            TransformedDataset(data_train),
            batch_size=list_learning_sample_size
        )
        table = [
            ["Algorithm", "Sample Size", "Sample Dimension", "Data Device", "Sparsity", "Margin", "Max Clusters"],
            ["Robust List Learning", list_learning_sample_size, data_train.shape[1] - 1, data_train.device, self.sparsity, self.margin, self.num_cluster + 1]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        sparse_classifier_clusters = robust_list_learner(
            next(iter(rl_dataloader))
        )

        # Perform conditional learning
        print(" ".join([self.header, "initializing conditional classification learner for homogeneous halfspaces ..."]))
        conditional_learner = ConditionalLearnerForFiniteClass(
            prev_header=self.header + ">",
            dim_sample = data_train.shape[1] - 1,
            num_iter=self.num_iter, 
            lr_coeff=self.lr_coeff, 
            sample_size_psgd=int(data_train.shape[0] * self.data_frac_psgd), 
            batch_size=self.batch_size
        ).to(data_train.device)

        table = [
            ["Algorithm", "Sample Size", "Sample Dimension", "Data Device", "Max Iterations", "LR Scaler", "Batch Size"],
            ["Conditional Classification", data_train.shape[0], data_train.shape[1] - 1, data_train.device, self.num_iter, self.lr_coeff, self.batch_size]
        ]

        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        classifier, selector = conditional_learner(
            data=data_train,
            sparse_classifier_clusters=sparse_classifier_clusters
        )

        print(" ".join([self.header, "finding empirical error minimizer from sparse perceptrons ..."]))
        eem_classifier, min_error = None, 1
        for classifiers in tqdm(
            sparse_classifier_clusters, 
            total=len(sparse_classifier_clusters), 
            desc=f"{self.header} - find EEM"
        ):
            error_rates, _, _ = self.error_rate_est(
                data_test=data_test,
                predict=SparseClassify,
                classifier=classifiers
            )
            error_rate, index = torch.min(error_rates, dim=0)
            if error_rate < min_error:
                min_error = error_rate
                eem_classifier = classifiers[index].to_dense()
        
        print(" ".join([self.header, "best classifier w/o selector is:", "\n",str(eem_classifier)]))
        print(" ".join([self.header, "best classifier with selector is:", "\n", str(classifier)]))
        print(" ".join([self.header, "best selector is:", "\n", str(selector)]))
        errorwo, error, coverage = self.error_rate_est(
            data_test=data_test, 
            predict=Classify, 
            classifier=classifier, 
            selector=selector
        )
        table = [
            ["Sample Size", "Sample Dimension", "Data Device", "Min ER w/o Selector", "Min ER with Selector","ER w/o Selector", "Coverage"],
            [data_test.shape[0], data_test.shape[1] - 1, data_test.device, min_error, error, errorwo, coverage]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def error_rate_est(
            self,
            data_test: torch.Tensor,
            predict: Callable,
            classifier: Union[torch.Tensor, torch.sparse.FloatTensor],
            selector: torch.Tensor = None
    ) -> Tuple[torch.float, torch.float, torch.float]:
        dataset_test = TransformedDataset(data_test)
        labels, features = dataset_test[:]

        errors = (
            predict(
                classifier=classifier,
                data=features.T
            )
        ) != labels  # [num_classifier, test data size]
        classification_ER = errors.sum(dim=-1)/len(dataset_test)

        conditional_classification_ER = classification_ER
        coverage = 1
        
        if isinstance(selector, torch.Tensor):
            selections = Classify(
                classifier=selector,
                data=features.T
            )    # [test data size]
            num_selection = selections.sum()
            # make sure it is not NaN
            conditional_classification_ER = (errors & selections).sum() / num_selection
            coverage = num_selection / len(dataset_test)

        return classification_ER, conditional_classification_ER, coverage