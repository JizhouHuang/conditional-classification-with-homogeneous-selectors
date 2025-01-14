import torch
import torch.nn as nn

class RobustListLearner(nn.Module):
    def __init__(self, sparsity, margin):
        """
        sparsity (int):           The degree for each combination.
        margin (int):             The margin to use for the robust list learning.
        """
        super(RobustListLearner, self).__init__()
        self.sparsity = sparsity
        self.margin = margin

    def forward(self, distr):
        """
        Perform robust list learning as specified by the algorithm in Appendix A.
        
        Parameters:
        distr (Dataset):                   The input data distribution. 
                                           The first column is the label, which takes values in {0, 1}.
        
        Returns:
        sparse_weight_list (torch.Tensor): The list of weights for each combination.
                                           The weight_list is represented as a sparse tensor.
                                           The order of the weight vectors in the list is the same as the following two loops:
                                           for features in feature_combinations:
                                               for samples in sample_combinations:
                                                   ...
        """
        

        # Extract features and labels
        # Assume the first column is the label column
        # and the rest are feature columns
        labels = distr[:, 0] * 2 - 1
        labeled_features = labels.unsqueeze(1) * distr[:, 1:]

        sample_size, sample_dim = labeled_features.shape[0], labeled_features.shape[1]
        
        # Generate row indices of all possible combinations of samples
        # dimension: [sample_size choose sparsity, sparsity]
        sample_indices = torch.combinations(torch.arange(sample_size), self.sparsity)
        num_sample_combinations = sample_indices.shape[0]

        # Select the labels for the generated sample combinations
        # dimension: [sample_size choose sparsity, sparsity]
        label_combinations = torch.index_select(
            labels,
            0,
            sample_indices.flatten()
        ).reshape(-1, self.sparsity)

        # Select the rows for the generated sample combinations while remaining flattened
        # dimension: [sample_size choose sparsity * sparsity, sample_dim]
        labeled_feature_combinations = torch.index_select(
            labeled_features,
            0,
            sample_indices.flatten()
        )

        # Generate column indices of all possible combinations of features
        # dimension: [sample_dim choose sparsity, sparsity]
        feature_indices = torch.combinations(torch.arange(sample_dim), self.sparsity)
        num_feature_combinations = feature_indices.shape[0]

        # Select the columns for the generated feature combinations from the flattened labeled_feature_combinations
        # dimension: [sample_dim choose sparsity, sparsity, (sample_size choose sparsity) * sparsity]
        labeled_feature_combinations = torch.t(
            torch.index_select(
            labeled_feature_combinations,
            1,
            feature_indices.flatten()
            )
        ).reshape(-1, self.sparsity, num_sample_combinations * self.sparsity)

        # dimension: [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, sparsity]
        labeled_feature_combinations = torch.transpose(
            labeled_feature_combinations, 
            1, 
            2
        ).reshape(-1, self.sparsity, self.sparsity)

        # repeat label_combinations to match the shape of labeled_feature_combinations
        # dimension: [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, 1]
        label_combinations = label_combinations.repeat(num_feature_combinations, 1)

        # solve the linear system specified in Algorithm 4 in Appendix A
        # weight_list = torch.linalg.solve(
        #     labeled_feature_combinations, 
        #     label_combinations - self.margin
        # )
        weight_list = torch.matmul(
            labeled_feature_combinations, 
            (label_combinations - self.margin).unsqueeze(-1)
        ).squeeze()

        # encoded as sparse vectors
        # construct a 2D coordinate tensor to indicate the position of non-zero elements
        row_indices = torch.arange(num_feature_combinations * num_sample_combinations).repeat_interleave(self.sparsity)
        col_indices = feature_indices.repeat_interleave(num_sample_combinations, dim=0).flatten()
        indices = torch.stack((row_indices, col_indices))

        # flatten the weight_list to match the shape of indices
        values = weight_list.flatten()
        # construct the size of the sparse tensor
        size = torch.Size([num_feature_combinations * num_sample_combinations, sample_dim])

        sparse_weight_list = torch.sparse_coo_tensor(indices, values, size)

        return sparse_weight_list

    # verification function
    def forward_verifier(self, distr):
        """
        Perform robust list learning of sparse linear classifiers for verification purpose.
        
        Parameters:
        distr (Dataset):            The input data distribution.
                                    The first column is the label, which takes values in {0, 1}.
        
        Returns:
        weight_list (torch.Tensor): The list of weights for each combination.
        """

        # Extract features and labels
        # Assume the first column is the label column
        # and the rest are feature columns
        labels = distr[:, 0] * 2 - 1
        labeled_features = labels.unsqueeze(1) * distr[:, 1:]

        sample_size, sample_dim = labeled_features.shape[0], labeled_features.shape[1]
        
        # Generate row indices of all possible combinations of samples
        # dimension: [sample_size choose sparsity, sparsity]
        sample_indices = torch.combinations(torch.arange(sample_size), self.sparsity)

        # Generate column indices of all possible combinations of features
        # dimension: [sample_dim choose sparsity, sparsity]
        feature_indices = torch.combinations(torch.arange(sample_dim), self.sparsity)


        weight_list = torch.zeros(sample_indices.shape[0] * feature_indices.shape[0],  sample_dim)
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
                weight_list[i, fid] = torch.mv(labeled_feature_combination_inv, label_combination - self.margin)
                i += 1

        return weight_list

# Test the function
# num_samples = 64
# sparsity = 3
# margin = 0.00238678712
# fake_distr = fake_data(num_samples, 5)
# robust_list_learner = RobustListLearner(sparsity, margin)
# weight_list = robust_list_learner.forward(fake_distr)
# weight_list_abs_sum = torch.abs(weight_list.sum())
# print(weight_list.shape, weight_list_abs_sum)

# weight_list_base = robust_list_learner.forward_verifier(fake_distr)
# weight_list_base_abs_sum = torch.abs(weight_list_base.sum())
# print(weight_list_base.shape, weight_list_base_abs_sum)

# diff = torch.abs(weight_list.to_dense() - weight_list_base).sum()
# print(diff)