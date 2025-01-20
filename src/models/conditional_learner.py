import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm
from typing import List, Tuple
from ..utils.data import TransformedDataset
from ..utils.predictions import LinearModel, ConditionalLinearModel
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
            classifier_clusters: List[ConditionalLinearModel]
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
        classifier_clusters (List[ConditionalLinearModel]): The list of sparse classifiers.

        Returns:
        selector_list (torch.Tensor): The list of weights for each classifier.
                                      The weight_list is represented as a sparse tensor.
                                      The order of the weight vectors in the list is the same as the following two loops:
                                      for features in feature_combinations:
                                          for samples in sample_combinations:
                                              ...
        """        
        
        candidate_selectors = torch.zeros(
            [len(classifier_clusters), self.dim_sample]
        ).to(data.device)
        candidate_classifiers = torch.zeros(
            [len(classifier_clusters), self.dim_sample]
        ).to(data.device)

        # initialize evaluation dataset for conditional learner
        labels_eval, features_eval = data[:, 0], data[:, 1:]

        for i, classifiers in enumerate(
            tqdm(
                classifier_clusters, 
                desc=self.header,
                # leave=False
            )
        ):
            dataset = TransformedDataset(
                data=data,
                predictor=classifiers
            )
            dataset_train, dataset_val = random_split(
                dataset, 
                [self.sample_size_psgd, len(dataset) - self.sample_size_psgd]
            )

            selector_learner = SelectorPerceptron(
                prev_header=self.header + ">",
                dim_sample=self.dim_sample,
                cluster_id = i + 1,
                cluster_size=classifiers.predictor.size(0),
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

            classifiers.set_selector(weights=selectors)
            _, min_ids = torch.min(
                classifiers.conditional_error_rate(
                    X=features_eval,
                    y=labels_eval
                ),
                dim=0
            )
            candidate_classifiers[i, ...], candidate_selectors[i, ...] = classifiers.predictor[min_ids].to_dense(), selectors[min_ids]
            classifiers.set_selector(weights=None)

        classifiers = ConditionalLinearModel(
            seletor_weights=candidate_selectors,
            predictor_weights=candidate_classifiers
        )
        _, min_ids = torch.min(
            classifiers.conditional_error_rate(
                X=features_eval,
                y=labels_eval
            ),
            dim=0
        )
        return ConditionalLinearModel(
            seletor_weights=candidate_selectors[min_ids],
            predictor_weights=candidate_classifiers[min_ids]
        )