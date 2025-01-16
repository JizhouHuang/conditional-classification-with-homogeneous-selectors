import torch
import torch.nn as nn
from ..utils.helpers import Classify, FixedIterationLoader
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

class SelectorPerceptron(nn.Module):
    def __init__(
            self, 
            prev_header: str,
            dim_sample: int, 
            cluster_id: int,
            cluster_size: int, 
            num_iter: int, 
            lr_beta: float, 
            batch_size: int,
            device: torch.device
        ):
        """
        Initialize the selector perceptron.

        Parameters:
        
        dataset (TransformedDataset):   The dataset to be used.
        cluster_size (int):             The number of classifiers.
        num_iter (int):                 The number of iterations for SGD.
        lr_coeff (float):               The learning rate coefficient.
        train_ratio (float):            The ratio of training samples.
        batch_size (int):               The batch size for SGD.
        """
        super(SelectorPerceptron, self).__init__()

        # Initialization
        self.header = " ".join([prev_header, "PSGD optim on cluster", str(cluster_id), "-"])
        self.dim_sample = dim_sample
        self.cluster_size = cluster_size
        self.num_iter = num_iter
        self.batch_size = batch_size

        # store the best selectors
        self.selector_list = torch.zeros(
            [
                2,
                self.cluster_size, 
                self.dim_sample
            ]
        ).to(device) # [2, cluster_size, dim_sample]

        # record the conditional error of the corresponding best selectors
        self.min_error = torch.ones(
            [
                2,
                self.cluster_size
            ]
        ).to(device) # [2, cluster_size]

        # learning rate
        self.beta = lr_beta

    def forward(
            self, 
            dataset_train: Subset,
            dataset_val: Subset,
            init_weight: torch.Tensor
        ) -> torch.Tensor:        

        # Create the dataloader for training
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=self.batch_size, 
            shuffle=True,   # change to False for DEBUG
            # num_workers=4,
            # pin_memory=True
        )

        # Evaluate the selector perceptron
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=len(dataset_val)
        )

        init_weight = init_weight.repeat(self.cluster_size, 1)   # [cluster_size, dim_sample]
        self.projected_SGD_alt(
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            weight_w=torch.stack([init_weight, -init_weight])    # [2, cluster_size, dim_sample]
        )

        self.pairwise_update(
            curr_error=self.min_error[0],
            min_error=self.min_error[1],
            curr_weight=self.selector_list[0],
            min_weight=self.selector_list[1]
        )

        return self.selector_list
    
    def projected_SGD_alt(
            self, 
            dataloader_train: DataLoader, 
            dataloader_val: DataLoader,
            weight_w: torch.Tensor  # [2, cluster_size, dim_sample]
        ) -> None:
        """
        Perform projected stochastic gradient descent.
        
        Parameters:
        dataloader_train (DataLoader):  The dataloader for the training dataset.
        dataloader_val (DataLoader):    The dataloader for the evaluation dataset.
        init_weight (torch.Tensor):     The initial weights for SGD.
        """

        # Initialize the list of weights
        dataset_val = next(iter(dataloader_val))
        dataloader_fixed = FixedIterationLoader(
            dataloader=dataloader_train,
            max_iterations=self.num_iter
        )
        # for i in tqdm(range(self.num_iter), desc="PSGD on GPU"):
        for data in tqdm(
            dataloader_fixed, 
            total=dataloader_fixed.max_iterations, 
            desc=self.header,
            leave=False
        ):
            # labels:   [data_batch_size, cluster_size]
            # features: [data_batch_size, dim_sample]
            labels, features = data

            # compute the product of label and indicator
            # print(f"{self.header} computing the selected labels ...")
            selected_labels = labels.T * (
                Classify(
                    classifier=weight_w,
                    data=features.T
                )
            )    # [2, cluster_size, data_batch_size]

            # project the features onto the orthogonal complement of the weight vector
            # print(f"{self.header} computing the orthogonal projection ...")
            orthogonal_projections = features - (weight_w @ features.T).unsqueeze(-1) * features # [2, cluster_size, data_batch_size, dim_sample]

            # print(f"{self.header} computing the gradients ...")
            gradients = selected_labels.unsqueeze(-1) * orthogonal_projections   # [2, cluster_size, data_batch_size, dim_sample]

            # update weights by average of gradients over all data samples (subject to change)
            # print(f"{self.header} computing gradient step ...")
            weight_u = weight_w - self.beta * torch.mean(gradients, dim=2)   # [2, cluster_size, dim_sample]
            # print(f"{self.header} normalizing weight vectors ...")
            weight_w = weight_u / torch.norm(weight_u, p=2, dim=-1).unsqueeze(-1)  # [cluster_size, dim_sample]

            # print(f"{self.header} starting updating the best weights ...")
            self.update_weight(
                dataset=dataset_val,
                weight_w=weight_w
            )
            torch.cuda.synchronize()

    def update_weight(
            self,
            dataset: List[torch.Tensor],
            weight_w: torch.Tensor
    ) -> None:
        """
        Update selector weights according to current classification error.

        Parameters:
        dataset (List[torch.Tensor]):   The evaluation dataset.
        weight_w (torch.Tensor):        The current weights for the selectors.
        """

        # labels:   [data_batch_size, cluster_size]
        # features: [data_batch_size, dim_sample]
        labels, features = dataset

        # compute the product of label and indicator for all samples
        # print(f"{self.header}> updating - computing current prediction ...")
        predictions = Classify(
            classifier=weight_w,
            data=features.T
        )  # [2, cluster_size, data_batch_size]
        # compute the conditional errors
        # print(f"{self.header}> updating - computing conditional errors ...")
        conditional_errors = (predictions.float() * labels.T).sum(dim=-1) / predictions.sum(dim=-1) # [2, cluster_size]
        # replace NaN to 1
        # print(f"{self.header}> updating - handling NaN values ...")
        conditional_errors[torch.isnan(conditional_errors)] = 1
        # update the weight list with those better selectors
        self.pairwise_update(
            curr_error=conditional_errors,
            min_error=self.min_error,
            curr_weight=weight_w,
            min_weight=self.selector_list
        )
    
    def pairwise_update(
            self,
            curr_error: torch.Tensor,
            min_error: torch.Tensor,
            curr_weight: torch.Tensor,
            min_weight: torch.Tensor
    ) -> None:
        # print(f"{self.header}> updating - computing indices for weights that need to update ...")
        indices = curr_error <= min_error   # [..., cluster_size]
        # print(f"{self.header}> updating - updating errors ...")
        self.min_error = min_error * ~indices + curr_error * indices   # [..., cluster_size]
        # print(f"{self.header}> updating - updating weights ...")
        self.selector_list = min_weight * ~indices.unsqueeze(-1) + curr_weight * indices.unsqueeze(-1) # [..., cluster_size, dim_sample]
        # print(f"{self.header}> updating - end")
