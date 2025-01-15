import torch
import torch.nn as nn
from ..utils.helpers import TransformedDataset
from typing import List
from torch.utils.data import DataLoader, random_split

class SelectorPerceptron(nn.Module):
    def __init__(
            self, 
            dataset: TransformedDataset, 
            num_classifier: int, 
            num_iter: int, 
            lr_coeff: float = 0.5, 
            train_ratio: float = 0.8, 
            batch_size: int = 32
        ):
        """
        Initialize the selector perceptron.

        Parameters:
        
        dataset (TransformedDataset):   The dataset to be used.
        num_classifier (int):           The number of classifiers.
        num_iter (int):                 The number of iterations for SGD.
        lr_coeff (float):               The learning rate coefficient.
        train_ratio (float):            The ratio of training samples.
        batch_size (int):               The batch size for SGD.
        """
        super(SelectorPerceptron, self).__init__()

        # Initialization
        self.dataset = dataset
        self.num_classsifier = num_classifier
        self.dim_sample = dataset.dim()
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        # store the best selectors
        self.selector_list = torch.zeros(
            [
                self.num_classsifier, 
                self.dim_sample
            ]
        ).to(dataset.device) # [num_classifier, dim_sample]

        # record the conditional error of the corresponding best selectors
        self.min_error = torch.ones(
            self.num_classsifier
        ).to(dataset.device) # [num_classifier]

        # learning rate
        self.beta = torch.sqrt(
            torch.tensor(
                lr_coeff/(self.num_iter * self.dim_sample)
            )
        )
    def forward(
            self, 
            init_weight: torch.Tensor
        ) -> torch.Tensor:
        # Split the dataset into training and evaluation sets
        train_size = int(self.train_ratio * len(self.dataset))
        eval_size = len(self.dataset) - train_size
        train_dataset, eval_dataset = random_split(
            self.dataset, 
            [train_size, eval_size]
        )

        # Create the dataloader for training
        dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True   # change to False for DEBUG
        )

        # Evaluate the selector perceptron
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=len(eval_dataset)    # may change to smaller batch size
        )

        for w in [init_weight, -init_weight]:
            self.projected_SGD_alt(
                dataloader=dataloader,
                eval_dataloader=eval_dataloader,
                init_weight=w
            )

        return self.selector_list
    
    def projected_SGD_alt(
            self, 
            dataloader: DataLoader, 
            eval_dataloader: DataLoader,
            init_weight: torch.Tensor
        ) -> None:
        """
        Perform projected stochastic gradient descent.
        
        Parameters:
        dataloader (DataLoader):            The dataloader for the training dataset.
        eval_dataset (DataLoader):          The dataloader for the evaluation dataset.
        init_weight (torch.Tensor):         The initial weights for SGD.
        """

        # Initialize the list of weights
        weight_w = init_weight.repeat(self.num_classsifier, 1)  # [num_classifier, dim_sample]
        eval_dataset = next(iter(eval_dataloader))
        i = 0
        while i < self.num_iter:
            for data in dataloader:

                # labels:   [data_batch_size, num_classifier]
                # features: [data_batch_size, dim_sample]
                labels, features = data

                # compute the product of label and indicator
                selected_labels = labels.T * (torch.matmul(weight_w, features.T) >= 0)    # [num_classifier, data_batch_size]

                # project the features onto the orthogonal complement of the weight vector
                orthogonal_projections = features - (weight_w @ features.T).unsqueeze(-1) * features # [num_classifier, data_batch_size, dim_sample]

                gradients = selected_labels.unsqueeze(-1) * orthogonal_projections   # [num_classifier, data_batch_size, dim_sample]

                # update weights by average of gradients over all data samples (subject to change)
                weight_u = weight_w - self.beta * torch.mean(gradients, dim=1)   # [num_classifier, dim_sample]
                weight_w = weight_u / torch.norm(weight_u, dim=1).unsqueeze(-1)  # [num_classifier, dim_sample]

                self.update_weights(
                    dataset=eval_dataset,
                    weight_w=weight_w
                )

                i += 1


    def update_weights(
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

        # labels:   [data_batch_size, num_classifier]
        # features: [data_batch_size, dim_sample]
        labels, features = dataset

        # compute the product of label and indicator for all samples
        predictions = torch.matmul(weight_w, features.T) >= 0  # [num_classifier, data_batch_size]
        # compute the conditional errors
        conditional_errors = (predictions.float() * labels.T).sum(dim=-1) / predictions.sum(dim=-1) # [num_classifier]
        # update the weight list with those better selectors
        indices = conditional_errors < self.min_error
        self.min_error[indices] = conditional_errors[indices]
        self.selector_list[indices, ...] = weight_w[indices]