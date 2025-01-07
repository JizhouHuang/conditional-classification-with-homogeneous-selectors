from ..utils.helpers import load_data, clean_data, fake_data
import torch

def robust_list_learning_of_sparse_linear_classifiers(file_path, num_samples, sparsity, margin, test_distr=None):
    """
    Perform robust list learning as specified by the algorithm in Appendix A.
    
    Parameters:
    file_path (str):            The path to the input data file.
    num_samples (int):          The number of samples to sample from data distribution.
    sparsity (int):             The degree for each combination.
    margin (int):               The margin to use for the robust list learning.
    test_distr (torch.Tensor):  The test distribution to use, where the labels are 0 or 1.
    
    Returns:
    weight_list (torch.Tensor): The list of weights for each combination.
                                The order of the weight vectors in the list is the same as the following two loops:
                                for features in feature_combinations:
                                    for samples in sample_combinations:
                                        ...
    """
    # Load and clean data using helpers
    if not file_path:
        # Generate fake data with 5 features
        distr = test_distr
    else:
        distr = load_data(file_path, k=num_samples)
        clean_data_df = clean_data(distr)

    # Extract features and labels
    # Assume the last column is the label column
    # and the rest are feature columns
    labels = distr[:, -1]*2 - 1
    labeled_features = labels.unsqueeze(1) * distr[:, :-1]

    sample_size, sample_dim = labeled_features.shape[0], labeled_features.shape[1]
    
    # Generate row indices of all possible combinations of samples
    # dimension: [sample_size choose sparsity, sparsity]
    sample_indices = torch.combinations(torch.arange(sample_size), sparsity)

    # Select the labels for the generated sample combinations
    # dimension: [sample_size choose sparsity, sparsity]
    label_combinations = torch.index_select(
        labels,
        0,
        sample_indices.flatten()
    ).reshape(-1, sparsity)

    # Select the rows for the generated sample combinations while remaining flattened
    # dimension: [sample_size choose sparsity * sparsity, sample_dim]
    labeled_feature_combinations = torch.index_select(
        labeled_features,
        0,
        sample_indices.flatten()
    )

    # Generate column indices of all possible combinations of features
    # dimension: [sample_dim choose sparsity, sparsity]
    feature_indices = torch.combinations(torch.arange(sample_dim), sparsity)

    # Select the columns for the generated feature combinations from the flattened labeled_feature_combinations
    # dimension: [sample_dim choose sparsity, sparsity, (sample_size choose sparsity) * sparsity]
    labeled_feature_combinations = torch.t(
        torch.index_select(
            labeled_feature_combinations,
            1,
            feature_indices.flatten()
        )
    ).reshape(-1, sparsity, labeled_feature_combinations.shape[0])

    # dimension: [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, sparsity]
    labeled_feature_combinations = torch.transpose(
        labeled_feature_combinations, 
        1, 
        2
    ).reshape(-1, sparsity, sparsity)

    label_combinations_with_margin = label_combinations - margin

    # dimension: [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, 1]
    label_combinations_with_margin = label_combinations_with_margin.repeat(feature_indices.shape[0], 1).view(-1, sparsity, 1)

    weight_list = torch.bmm(
        torch.linalg.inv(labeled_feature_combinations), 
        label_combinations_with_margin
    )

    return weight_list.squeeze(2)

# checker function
def robust_list_learning_of_sparse_linear_classifiers_checker(file_path, num_samples, sparsity, margin, test_distr=None):
    """
    Perform robust list learning of sparse linear classifiers.
    
    Parameters:
    file_path (str):            The path to the input data file.
    num_samples (int):          The number of samples to sample from data distribution.
    sparsity (int):             The degree for each combination.
    margin (int):               The margin to use for the robust list learning.
    test_distr (torch.Tensor):  The test distribution to use.
    
    Returns:
    weight_list (torch.Tensor): The list of weights for each combination.
    """
    # Load and clean data using helpers
    if not file_path:
        # Generate fake data with 5 features
        distr = test_distr
    else:
        distr = load_data(file_path, k=num_samples)
        clean_data_df = clean_data(distr)

    # Extract features and labels
    # Assume the last column is the label column
    # and the rest are feature columns
    labels = distr[:, -1] * 2 - 1
    labeled_features = labels.unsqueeze(1) * distr[:, :-1]

    sample_size, sample_dim = labeled_features.shape[0], labeled_features.shape[1]
    
    # Generate row indices of all possible combinations of samples
    # dimension: [sample_size choose sparsity, sparsity]
    sample_indices = torch.combinations(torch.arange(sample_size), sparsity)

    # Generate column indices of all possible combinations of features
    # dimension: [sample_dim choose sparsity, sparsity]
    feature_indices = torch.combinations(torch.arange(sample_dim), sparsity)


    weight_list = torch.zeros(sample_indices.shape[0] * feature_indices.shape[0],  sparsity)
    i = 0
    for fid in feature_indices:
        for sid in sample_indices:
            labeled_feature_combination = torch.index_select(
                torch.index_select(
                    labeled_features,
                    1,
                    fid
                ),
                0, 
                sid
            )
            labeled_feature_combination_inv = torch.linalg.inv(labeled_feature_combination)
            label_combination = torch.index_select(
                labels,
                0,
                sid
            )
            weight_list[i, :] = torch.mv(labeled_feature_combination_inv, label_combination - margin)
            i += 1

    return weight_list

# Test the function
num_samples = 1000
sparsity = 3
margin = 0.12
fake_distr = fake_data(num_samples, 5)
weight_list = robust_list_learning_of_sparse_linear_classifiers(None, num_samples, sparsity, margin, test_distr=fake_distr)
print(weight_list.shape)

# weight_list_base = robust_list_learning_of_sparse_linear_classifiers_checker(None, num_samples, sparsity, margin, test_distr=fake_distr)
# print(weight_list_base.shape)

# diff = torch.abs(weight_list - weight_list_base).sum()
# print(diff)