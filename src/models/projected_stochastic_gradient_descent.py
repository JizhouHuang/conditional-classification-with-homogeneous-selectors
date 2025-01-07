import torch
from ..utils.helpers import load_data, fake_data

def projected_sgd(data, num_samples, num_iters, init_weight):
    """
    Perform projected stochastic gradient descent.
    
    Parameters:
    data (pd.DataFrame): The input data for SGD.
                         [num_samples, num_features+1] where the last column is the label.
    num_samples (int): The number of samples to sample from data distribution.
    num_iters (int): The number of iterations for SGD.
    init_weight (torch.Tensor): The initial weights for SGD.
    
    Returns:
    weight_list (torch.Tensor): The list of weights for each iteration.
    """
    # Implement the algorithm here
    labels = data[:, -1]
    features = data[:, :-1]
    param_beta = torch.sqrt(
        torch.tensor(
            1/(num_iters * features.shape[1])
        )
    )

    weight_list = torch.zeros([num_iters, features.shape[1]])
    weight_list[0, :] = init_weight
    weight_w = init_weight
    for i in range(num_iters):
        # compute the product of label and indicator for all samples
        selected_labels = labels * (torch.mv(features, weight_w) >= 0)

        # project the features onto the orthogonal complement of the weight vector
        projected_features = torch.matmul(features, torch.eye(features.shape[1]) - torch.outer(weight_w,weight_w))

        gradient = selected_labels.unsqueeze(1) * projected_features

        # update weights by average of gradients over all data samples (subject to change)
        weight_u = weight_w - param_beta * torch.mean(gradient, dim=0)
        weight_w = weight_u / torch.norm(weight_u)

        # store weights
        weight_list[i, :] = weight_w

    return weight_list

# Test the function
num_samples = 1000
dim_sample = 5
num_iters = 100
fake_distr = fake_data(num_samples, dim_sample)
init_weight = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
weight_list = projected_sgd(fake_distr, num_samples, num_iters, init_weight)
print(weight_list.shape)