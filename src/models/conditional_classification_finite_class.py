import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import List, Tuple
from ..utils.helpers import TransformedDataset, LabelMapping
from ..models.projected_stochastic_gradient_descent import SelectorPerceptron

class ConditionalLearnerForFiniteClass(nn.Module):
    def __init__(
            self, 
            dim_sample: int,
            num_iter: int, 
            lr_coeff: float = 0.5, 
            train_ratio: float = 0.8, 
            batch_size: int = 32
    ):
        """
        Initialize the conditional learner for finite class classification.

        Parameters:
        dim_sample (int):             The dimension of the sample features.
        num_iter (int):               The number of iterations for SGD.
        lr_coeff (float):             The learning rate coefficient.
        train_ratio (float):          The ratio of training samples.
        batch_size (int):             The batch size for SGD.
        """
        super(ConditionalLearnerForFiniteClass, self).__init__()
        self.dim_sample = dim_sample
        self.num_iter = num_iter
        self.lr_coeff = lr_coeff
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        self.init_weight = torch.zeros(self.dim_sample, dtype=torch.float32)
        self.init_weight[0] = 1

    def forward(
            self, 
            data: torch.Tensor,
            sparse_classifier_clusters: List[torch.sparse.FloatTensor]
    ) -> torch.Tensor:
        """
        Perform conditional learning for finite class classification.


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
        
        print("conditional learner - initiating candidate selectors and classifiers ...")
        candidate_selectors = torch.zeros(
            [len(sparse_classifier_clusters), self.dim_sample]
        )
        candidate_classifiers = torch.zeros(
            [len(sparse_classifier_clusters), self.dim_sample]
        )
        print("conditional learner - start iterating sparse classifiers ...")
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     pool.map(
        #         self.parallel_subroutine, 
        #         [(i, sparse_classifier_clusters[i]) for i in range(len(sparse_classifier_clusters))]
        #     )

        for i, classifiers in enumerate(sparse_classifier_clusters):
            print(f"conditional learner - starting PSGD for the {i + 1}th cluster ...")
            selector_learner = SelectorPerceptron(
                dataset=TransformedDataset(data, LabelMapping(classifiers)),
                num_classifier=classifiers.size(0),
                num_iter=self.num_iter,
                lr_coeff=self.lr_coeff,
                train_ratio=self.train_ratio,
                batch_size=self.batch_size
            )
            selectors = selector_learner(
                self.init_weight.to(data.device)
            )  # [cluster size, dim sample]

            print(f"conditional learner - evaluating for the {i + 1}th cluster ...")
            candidate_classifiers[i, ...], candidate_selectors[i, ...] = self.evaluate(
                eval_dataset=TransformedDataset(data),   # could use different data for each evaluation
                classifiers=classifiers.to_dense(),
                selectors=selectors
            )

        print(f"conditional learner - evaluating for the final candidates...")
        return self.evaluate(
            eval_dataset=TransformedDataset(data),
            classifiers=candidate_classifiers,
            selectors=candidate_selectors
        )

    def evaluate(
            self,
            eval_dataset: TransformedDataset,
            classifiers: torch.Tensor,
            selectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels, features = eval_dataset[:]

        errors = (classifiers @ features.T >= 0) != labels
        selections = torch.matmul(selectors, features.T) >= 0

        conditional_error_rate = (errors * selections).sum(dim=-1) / selections.sum(dim=-1)
        # replace NaN to 1
        conditional_error_rate[torch.isnan(conditional_error_rate)] = 1

        min_error, min_index = torch.min(conditional_error_rate, dim=0)

        return classifiers[min_index], selectors[min_index]