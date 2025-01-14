import torch
import torch.nn as nn
from ..utils.helpers import UCIMedicalDataset, LabelMapping
from torch.utils.data import Dataset, DataLoader, random_split

class SelectorPerceptron(nn.Module):
    def __init__(self, dataset, num_classifier, num_iter, lr_coeff=0.5, train_ratio=0.8, batch_size=32):
        """
        Initialize the selector perceptron.

        Parameters:
        dim_sample (int):       The dimension of the sample.
        num_iter (int):         The number of iterations for SGD.
        lr_coeff (float):       The learning rate coefficient.
        train_ratio (float):    The ratio of training samples.
        batch_size (int):       The batch size for SGD.
        """
        super(SelectorPerceptron, self).__init__()

        # Initialization
        self.dataset = dataset
        self.num_classsifier = num_classifier
        self.dim_sample = dataset.dim()
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        # learning rate
        self.beta = torch.sqrt(
            torch.tensor(
                lr_coeff/(self.num_iter * self.dim_sample)
            )
        )
    def forward(self, init_weight):
        # Split the dataset into training and evaluation sets
        train_size = int(self.train_ratio * len(self.dataset))
        eval_size = len(self.dataset) - train_size
        train_dataset, eval_dataset = random_split(self.dataset, [train_size, eval_size])

        # Create the dataloader for training
        dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        selectors = torch.cat(
            (
                self.projected_SGD(dataloader, init_weight), 
                self.projected_SGD(dataloader, -init_weight)
            ), 
            dim=0
        )

        # Evaluate the selector perceptron
        dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
        return self.evaluate(next(iter(dataloader)), selectors)

    def projected_SGD(self, dataloader, init_weight):
        """
        Perform projected stochastic gradient descent.
        
        Parameters:
        dataset (Dataset):          A dataset with transformed labels.
                                    [batch_size, num_features+1] where the first column is the label.
        init_weight (torch.Tensor): The initial weights for SGD.

        Returns:
        torch.Tensor:               The list of weights for each iteration.
        """

        # Initialize the list of weights
        selector_list = torch.zeros([self.num_iter + 1, self.num_classsifier, self.dim_sample]) # [num_iter + 1, num_classifier, dim_sample]
        weight_w = init_weight.repeat(self.num_classsifier, 1)  # [num_classifier, dim_sample]
        selector_list[0, :, :] = weight_w 

        i = 0
        while i < self.num_iter:
            for data in dataloader:
                # Split the data into features and labels
                # labels:   [data_batch_size, num_classifier]
                # features: [data_batch_size, dim_sample]
                labels, features = data

                # compute the product of label and indicator
                selected_labels = labels.T * (torch.matmul(weight_w, features.T) >= 0)    # [num_classifier, data_batch_size]

                # project the features onto the orthogonal complement of the weight vector
                # projected_features = torch.matmul( 
                #     torch.eye(self.dim_sample) - outer_products,
                #     features.T
                # )   # [data_batch_size, dim_sample]
                orthogonal_projections = features - (weight_w @ features.T).unsqueeze(-1) * features # [num_classifier, data_batch_size, dim_sample]

                gradients = selected_labels.unsqueeze(-1) * orthogonal_projections   # [num_classifier, data_batch_size, dim_sample]

                # update weights by average of gradients over all data samples (subject to change)
                weight_u = weight_w - self.beta * torch.mean(gradients, dim=1)   # [num_classifier, dim_sample]
                weight_w = weight_u / torch.norm(weight_u, dim=1).unsqueeze(-1)  # [num_classifier, dim_sample]
                if i < self.num_iter:
                    selector_list[i + 1, :, :] = weight_w
                else:
                    break
                i += 1
        return selector_list 

    def evaluate(self, dataset, selector_list):
        """
        Evaluate the selector perceptron to find the best selector.

        Parameters:
        dataset (Dataset):              [num_eval_sample, num_features+1] where the first column is the label.
        selector_list (torch.Tensor):   The list of weights for each iteration.

        Returns:
        torch.Tensor:                   The best selector.
        """
        # Split the data into features and labels
        # labels:   [data_batch_size, num_classifier]
        # features: [data_batch_size, dim_sample]
        labels, features = dataset[:]

        # compute the product of label and indicator for all samples
        predictions = torch.matmul(selector_list, features.T) >= 0  # [num_iter + 1, num_classifier, data_batch_size]
        # compute the conditional errors
        conditional_errors = (predictions.float() * labels.T).sum(dim=-1) / predictions.sum(dim=-1) # [num_iter + 1, num_classifier]

        # the best selectors for every classifier
        best_selectors = torch.min(conditional_errors, dim=0)   # [num_classifier]

        best_weights = selector_list[best_selectors.indices, torch.arange(self.num_classsifier)]    # [num_classifier, dim_sample]

        return best_weights, best_selectors.values

# Load the dataset
file_path = "src/lib/hepatitis.data"  # Replace with your file path
column_names = [
    "Class",  # Target (DIE=1, LIVE=2)
    "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
    "Anorexia", "Liver Big", "Liver Firm", "Spleen Palpable", "Spiders",
    "Ascites", "Varices", "Bilirubin", "Alk Phosphate", "SGOT",
    "Albumin", "Protime", "Histology"
]
# categorical_attr = ["Sex"]

# batch_size = 70
# num_iters = 4096
# train_ratio = 0.8

# uci_dataset = UCIMedicalDataset(file_path, column_names, categorical_attr)
# data_dim = uci_dataset.dim()
# init_weight = torch.zeros(data_dim, dtype=torch.float32)
# init_weight[0] = -1
# label_mapping = LabelMapping(torch.randn(uci_dataset.dim()))
# # uci_dataset.set_transform(label_mapping)
# dloader = DataLoader(uci_dataset, batch_size=len(uci_dataset))
# errors = next(iter(dloader))[:, 0].sum() / len(uci_dataset)
# print(errors)
# # dataloader = DataLoader(uci_dataset, batch_size=batch_size, shuffle=True)

# # weight_list = projected_sgd(dataloader, num_iters, init_weight)
# # weight_list_sum = torch.abs(weight_list.sum())
# # print(weight_list.shape, weight_list_sum)


# sperceptron = SelectorPerceptron(uci_dataset, num_iters, train_ratio=train_ratio, batch_size=batch_size)
# weight, error = sperceptron(init_weight)
# print(error)
# print(weight)
# # diff = torch.abs(weight_list_torch - weight_list).sum()
# # print(diff)
# # diff_last = torch.abs(weight_list_torch[-1, :] - weight_list[-1, :]).sum()
# # print(torch.abs(weight_list_torch[-1, :]).sum(), torch.abs(weight_list[-1, :]).sum(), diff_last)