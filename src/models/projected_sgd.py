import torch
import torch.nn as nn
from ..utils.data import FixedIterationLoader
from ..utils.predictions import ConditionalLinearModel
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

class SelectorPerceptron(nn.Module):
    """
    Conditional Classification of Homogeneous Halfspaces using Projected Stochastic Gradient Descent.
    """
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
        self.device = device

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
            shuffle=True
        )

        # Evaluate the selector perceptron
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=len(dataset_val)
        )

        init_weight = init_weight.repeat(self.cluster_size, 1)   # [cluster_size, dim_sample]
        self.projected_SGD(
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            model=ConditionalLinearModel(
                seletor_weights=torch.stack(
                    [init_weight, -init_weight]
                ).to(self.device)   # [2, cluster_size, dim_sample]
            )
        )

        self.pairwise_update(
            curr_error=self.min_error[0],
            min_error=self.min_error[1],
            curr_weight=self.selector_list[0],
            min_weight=self.selector_list[1]
        )

        return self.selector_list
    
    def projected_SGD(
            self, 
            dataloader_train: DataLoader, 
            dataloader_val: DataLoader,
            model: ConditionalLinearModel
        ) -> None:
        """
        Perform projected stochastic gradient descent.
        
        Parameters:
        dataloader_train (DataLoader):  The dataloader for the training dataset.
        dataloader_val (DataLoader):    The dataloader for the evaluation dataset.
        model (LinearModel):            The initial weights for SGD.
        """

        # Initialize the list of weights
        labels_val, features_val = next(iter(dataloader_val))
        dataloader_fixed = FixedIterationLoader(
            dataloader=dataloader_train,
            max_iterations=self.num_iter
        )
        # enable to show progress bar
        # for data in tqdm(
        #     dataloader_fixed, 
        #     total=dataloader_fixed.max_iterations, 
        #     desc=self.header,
        #     leave=False
        # ):
        for data in dataloader_fixed:
            # labels:   [data_batch_size, cluster_size]
            # features: [data_batch_size, dim_sample]
            labels, features = data

            # gradient step
            model.selector.update(
                weights= - self.beta * model.selector.proj_grad(
                    X=features,
                    y=labels.t()
                )
            )
            # compute conditional error rates
            conditional_errors = model.conditional_error_rate(
                X=features_val,
                y=labels_val.t()
            )

            self.pairwise_update(
                curr_error=conditional_errors,
                min_error=self.min_error,
                curr_weight=model.selector.weights,
                min_weight=self.selector_list
            )
            # torch.cuda.synchronize()
    
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
