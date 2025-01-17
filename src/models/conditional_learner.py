import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm
from typing import List, Tuple
from ..utils.helpers import Classify, TransformedDataset
from .projected_sgd import SelectorPerceptron

class ConditionalLearnerForFiniteClass(nn.Module):
    """
    Conditional Classification for Any Finite Classes
    """
    def __init__(
            self, 
            prev_header: str,
            dim_sample: int,
            num_iter: int, 
            sample_size_psgd: int,
            lr_coeff: float = 0.5,
            batch_size: int = 32
    ):
        """
        Initialize the conditional learner for finite class classification.
        Compute the learning rate of PSGD for the given lr coefficient using
        the formula:
            beta = O(sqrt(1/num_iter * dim_sample)).

        Parameters:
        dim_sample (int):             The dimension of the sample features.
        num_iter (int):               The number of iterations for SGD.
        lr_coeff (float):             The learning rate coefficient.
        sample_size_psgd (float):          The ratio of training samples.
        batch_size (int):             The batch size for SGD.
        """
        super(ConditionalLearnerForFiniteClass, self).__init__()
        self.header = " ".join([prev_header, "conditional learner", "-"])
        self.dim_sample = dim_sample
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.sample_size_psgd = sample_size_psgd

        self.lr_beta = lr_coeff * torch.sqrt(
            torch.tensor(
                1 / (num_iter * dim_sample)
            )
        )
        self.init_weight = torch.zeros(self.dim_sample, dtype=torch.float32)
        self.init_weight[0] = 1

    def forward(
            self, 
            data: torch.Tensor,
            sparse_classifier_clusters: List[torch.sparse.FloatTensor]
    ) -> torch.Tensor:
        """
        Call PSGD optimizer for each cluster of sparse classifiers using all the data given.
        
        Note that the PSGD optimizer runs in parallel for all the sparse classifiers in a 
        cluster. PSGD optimizer will return one selector for each sparse classifier of each 
        cluster.

        For each cluster, we evaluate the best classifier-selector pair using all the data given
        due to insufficient data size.

        At last, we use the same data set to find the best classifier-selector pair across cluster.

        Parameters:
        sparse_classifier_clusters (List[torch.sparse.FloatTensor]): The list of sparse classifiers.

        Returns:
        selector_list (torch.Tensor): The list of weights for each classifier.
                                      The weight_list is represented as a sparse tensor.
                                      The order of the weight vectors in the list is the same as the following two loops:
                                      for features in feature_combinations:
                                          for samples in sample_combinations:
                                              ...
        """        
        
        num_cluster = len(sparse_classifier_clusters)
        candidate_selectors = torch.zeros(
            [num_cluster, self.dim_sample]
        ).to(data.device)
        candidate_classifiers = torch.zeros(
            [num_cluster, self.dim_sample]
        ).to(data.device)

        # initialize evaluation dataset for conditional learner
        eval_dataset = TransformedDataset(data)

        for i, classifiers in enumerate(
            tqdm(
                sparse_classifier_clusters, 
                # total=num_cluster, 
                desc=self.header,
                # leave=False
            )
        ):
            dataset = TransformedDataset(data, classifiers)
            dataset_train, dataset_val = random_split(
                dataset, 
                [self.sample_size_psgd, len(dataset) - self.sample_size_psgd]
            )
            selector_learner = SelectorPerceptron(
                prev_header=self.header + ">",
                dim_sample=self.dim_sample,
                cluster_id = i + 1,
                cluster_size=classifiers.size(0),
                num_iter=self.num_iter,
                lr_beta=self.lr_beta,
                batch_size=self.batch_size,
                device=data.device
            )
            selectors = selector_learner(
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                init_weight=self.init_weight.to(data.device)
            )  # [cluster size, dim sample]

            candidate_classifiers[i, ...], candidate_selectors[i, ...] = self.evaluate(
                eval_dataset=eval_dataset,   # could use different data for each evaluation
                classifiers=classifiers.to_dense(),
                selectors=selectors
            )

        print(f"{self.header} evaluating for the final candidates...")
        return self.evaluate(
            eval_dataset=eval_dataset,
            classifiers=candidate_classifiers,
            selectors=candidate_selectors
        )

    def evaluate(
            self,
            eval_dataset: TransformedDataset,
            classifiers: torch.Tensor,
            selectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the classifiers and selectors on the given evaluation dataset.

        This function computes the conditional error rates for all the classifier-selector pairs in
        a cluster and returns the pair with the minimum error rate.

        Parameters:
        eval_dataset (TransformedDataset): The dataset to evaluate the classifiers and selectors.
        classifiers (torch.Tensor): The tensor containing classifier weights.
        selectors (torch.Tensor): The tensor containing selector weights.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The classifier and selector with the minimum error rate.
        """
        labels, features = eval_dataset[:]

        errors = (
            Classify(
                classifier=classifiers,
                data=features.T
            )
        ) != labels
        selections = Classify(
            classifier=selectors,
            data=features.T
        )

        conditional_error_rate = (errors * selections).sum(dim=-1) / selections.sum(dim=-1)
        # replace NaN to 1
        conditional_error_rate[torch.isnan(conditional_error_rate)] = 1

        min_error, min_index = torch.min(conditional_error_rate, dim=0)

        return classifiers[min_index], selectors[min_index]