import torch
import torch.nn as nn
import yaml
from typing import Callable, Union, Tuple, List
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from ..utils.helpers import Classify, SparseClassify, TransformedDataset, UCIMedicalDataset
from ..models.conditional_learner import ConditionalLearnerForFiniteClass
from ..models.robust_list_learner import RobustListLearner

class ExperimentCCSC(nn.Module):
    """
    Experiment of Conditional Classification for Sparse Classes.
    """
    def __init__(
            self,
            prev_header: str,
            experiment_id: int,
            config_file_path: str
    ):
        """
        Initialize through reading parameters from YAML file located under src/config/.

        Parameters:
        experiment_id (int): The ID of the experiment.
        config_file_path (str): The path to the configuration file.

        Explanations:
        num_sample_rll:     Number of training data used for Robust List Learning.
        margin:             According to Appendix A, the RHS of the linear system is formed by labels subtracted by the margin.
        sparsity:           Number of non-zero dimensions for the resulting sparse representations.
        cluster_size:       To speed up the computation, instead of iterating only one classifier at a time, 
                            we partition all the sparse classifiers into multiple clusters and run PSGD on a cluster in each iteration.
        data_frac_psgd:     Fraction of training data used for the updating stage of Projected SGD.
        lr_coeff:           A constant to scale the learning rate (beta) of PSGD.
        num_iter:           The number of iteration for PSGD to run.
        batch_size:         Number of example to estimate the expectation of projected gradient in each gradient step.
        """
        super(ExperimentCCSC, self).__init__()
        self.header = " ".join([prev_header, "experiment", str(experiment_id), "-"])

        # Read the YAML configuration file
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Load configuration values
        self.data_frac = config['data_frac']
        self.num_sample_rll = config['num_sample_rll']
        self.margin = config['margin']
        self.sparsity = config['sparsity']
        self.cluster_size = config['cluster_size']
        self.data_frac_psgd = config['data_frac_psgd']
        self.lr_coeff = config['lr_coeff']
        self.num_iter = config['num_iter']
        self.batch_size = config['batch_size']

    def forward(
            self,
            data: UCIMedicalDataset
    ) -> List[
        Tuple[
            torch.Tensor, 
            Union[
                torch.float, 
                Tuple[torch.float, torch.float]
            ]
        ]
    ]:
        """
        Call Robust List Learner to generate a list of sparse classifiers and input them to the Conditional Learner.

        Parameters:
        data_train:     Training data for both Robust List Learning and Conditional Learning.
        data_test:      Testing data to estimate the error measures of the final classifier-selector pair.
                        Disjoint from data_train.
        """
        # Learn the sparse classifiers
        print(" ".join([self.header, "initializing robust list learner for sparse perceptrons ..."]))
        data_train, data_test = data.slice_with_ratio(self.data_frac)

        robust_list_learner = RobustListLearner(
            prev_header=self.header + ">",
            sparsity=self.sparsity, 
            margin=self.margin,
            cluster_size=self.cluster_size
        ).to(data_train.device)

        rl_dataloader = DataLoader(
            TransformedDataset(data_train),
            batch_size=self.num_sample_rll
        )

        sparse_classifier_clusters = robust_list_learner(
            next(
                iter(
                    DataLoader(
                        TransformedDataset(data_train),
                        batch_size=self.num_sample_rll
                    )
                )
            )
        )   # List[torch.sparse.FloatTensor]

        table = [
            ["Algorithm", "Sample Size", "Sample Dimension", "Data Device", "Sparsity", "Margin", "Cluster Size", "Max Clusters"],
            ["List Learning", min(self.num_sample_rll, data_train.size(0)), data_train.shape[1] - 1, data_train.device, min(data_train.shape[1] - 1, self.sparsity), self.margin, sparse_classifier_clusters[0].size(0), len(sparse_classifier_clusters)]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

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
            ["Cond Classification", data_train.shape[0], data_train.shape[1] - 1, data_train.device, self.num_iter, self.lr_coeff, self.batch_size]
        ]

        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        classifier, selector = conditional_learner(
            data=data_train,
            sparse_classifier_clusters=sparse_classifier_clusters
        )   # Tuple[torch.Tensor, torch.Tensor]

        # model selection for sparse classifiers based on regular classification error
        print(" ".join([self.header, "finding empirical error minimizer from sparse perceptrons ..."]))
        eem_classifier, min_error = None, 1
        for classifiers in tqdm(
            sparse_classifier_clusters, 
            total=len(sparse_classifier_clusters), 
            desc=f"{self.header} find EEM",
            leave=False
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
        
        # Estimate error measures with selectors
        errorwo, error, coverage = self.error_rate_est(
            data_test=data_test, 
            predict=Classify, 
            classifier=classifier, 
            selector=selector
        )

        res = [
            (eem_classifier, min_error),
            (classifier, errorwo),
            (torch.stack([classifier, selector]), (error, coverage))
        ]
        
        # Print the results in a table format
        table = [
            ["Classifier Type", "Test Sample Size", "Classifier Sparsity", "Est Error Rate", "Coverage"],
            ["Classic Sparse", data_test.shape[0], self.sparsity, min_error, 1],
            ["Cond Sparse w/o Selector", data_test.shape[0], self.sparsity, errorwo, 1],
            ["Cond Sparse", data_test.shape[0], self.sparsity, error, coverage]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        return res

    def error_rate_est(
            self,
            data_test: torch.Tensor,
            predict: Callable,
            classifier: Union[torch.Tensor, torch.sparse.FloatTensor],
            selector: torch.Tensor = None
    ) -> Tuple[torch.float, torch.float, torch.float]:
        """
        Estimating the classification error rate, conditional classification error rate, coverage.
        If no selector given, conditional classification ER will be the same as classification ER,
        and coverage will be 1.

        Parameters:
        data_test (torch.Tensor): The test data.
        predict (Callable): The prediction function.
        classifier (Union[torch.Tensor, torch.sparse.FloatTensor]): The classifier.
        selector (torch.Tensor, optional): The selector. Defaults to None.

        Returns:
        Tuple[torch.float, torch.float, torch.float]: The error rates and coverage.
        """
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