import torch
import torch.nn as nn
from ..utils.helpers import TransformedDataset

class RobustListLearner(nn.Module):
    def __init__(
            self, 
            sparsity: int, 
            margin: float,
            num_slices: int
    ):
        """
        sparsity (int):           The degree for each combination.
        margin (int):             The margin to use for the robust list learning.
        """
        super(RobustListLearner, self).__init__()
        self.sparsity = sparsity
        self.margin = margin
        self.num_slices = num_slices

    def forward(
            self, 
            dataset: TransformedDataset
    ) -> list[torch.sparse.FloatTensor]:
        """
        Perform robust list learning as specified by the algorithm in Appendix A.
        
        Parameters:
        dataset (torch.Tensor):              The input dataset. 
                                           The first column is the label, which takes values in {0, 1}.
        
        Returns:
        sparse_weight_list (list[torch.sparse.FloatTensor]): The list of weights for each combination.
                                           The weight_list is represented as a sparse tensor.
                                           The order of the weight vectors in the list is the same as the following two loops:
                                           for features in feature_combinations:
                                               for samples in sample_combinations:
                                                   ...
        """
        

        # Extract features and labels
        # Assume the first column is the label column
        # Map the labels from {0, 1} to {-1, +1}
        labels, features = dataset[:]
        print(f"robust list learner - is labels on CUDA? {labels.is_cuda}")
        labeled_features = labels.unsqueeze(1) * features

        sample_size, self.sample_dim = labeled_features.shape[0], labeled_features.shape[1]
        
        # Generate row indices of all possible combinations of samples
        sample_indices = torch.combinations(
            torch.arange(sample_size), 
            self.sparsity
        ).to(labels.device)   # [sample_size choose sparsity, sparsity]
        
        self.num_sample_combinations = sample_indices.shape[0]
        print(f"robust list learner - is sample indices on CUDA? {sample_indices.is_cuda}")

        # Generate column indices of all possible combinations of features
        feature_indices = torch.combinations(
            torch.arange(self.sample_dim), 
            self.sparsity
        ).to(labels.device)   # [sample_dim choose sparsity, sparsity]
        
        self.num_feature_combinations = feature_indices.shape[0]
        print(f"robust list learner - is feature indices on CUDA? {feature_indices.is_cuda}")

        # Select the labels for the generated sample combinations
        label_combinations = torch.index_select(
            labels,
            0,
            sample_indices.flatten()
        ).reshape(-1, self.sparsity)    # [sample_size choose sparsity, sparsity]

        # Select the rows for the generated sample combinations while remaining flattened
        labeled_feature_combinations = torch.index_select(
            labeled_features,
            0,
            sample_indices.flatten()
        )   # [sample_size choose sparsity * sparsity, sample_dim]

        # Select the columns for the generated feature combinations from the flattened labeled_feature_combinations
        labeled_feature_combinations = torch.t(
            torch.index_select(
            labeled_feature_combinations,
            1,
            feature_indices.flatten()
            )
        ).reshape(
            -1, 
            self.sparsity, 
            self.num_sample_combinations * self.sparsity
        )   # [sample_dim choose sparsity, sparsity, (sample_size choose sparsity) * sparsity]

        labeled_feature_combinations = torch.transpose(
            labeled_feature_combinations, 
            1, 
            2
        ).reshape(
            -1, 
            self.sparsity, 
            self.sparsity
        ) # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, sparsity]

        # repeat label_combinations to match the shape of labeled_feature_combinations
        label_combinations = label_combinations.repeat(
            self.num_feature_combinations, 
            1
        )   # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]

        ### solve the linear system specified in Algorithm 4 in Appendix A ###
        # weight_list = torch.linalg.solve(
        #     labeled_feature_combinations, 
        #     label_combinations - self.margin
        # )
        weight_list = torch.matmul(
            torch.linalg.pinv(labeled_feature_combinations), 
            (label_combinations - self.margin).unsqueeze(-1)
        ).squeeze() # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]

        # batch method
        sparse_weight_list = self.to_batched_sparse_tensor(
            weights=weight_list,
            feature_combinations=feature_indices
        )

        return sparse_weight_list

    def to_batched_sparse_tensor(
            self,
            weights: torch.Tensor,
            feature_combinations: torch.Tensor
    ) -> list[torch.sparse.FloatTensor]:

        col_indices = feature_combinations.repeat_interleave(
            self.num_sample_combinations,
            dim=0
        )   # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]

        batch_size = col_indices.shape[0] // self.num_slices

        row_indices = torch.arange(
            batch_size
        ).repeat_interleave(self.sparsity).to(col_indices.device)

        list_of_sparse_tensors = []
        for i in range(self.num_slices):
            print(f"robust list learner - computing the {i + 1}th cluster ...")
            list_of_sparse_tensors.append(
                self.to_sparse_tensor(
                    row_indices=row_indices,
                    col_indices=col_indices[i * batch_size : (i + 1) * batch_size],
                    weight_slice=weights[i * batch_size : (i + 1) * batch_size]
                )
            )
        if (i + 1) * batch_size < col_indices.shape[0]:
            print(f"robust list learner - computing the {i + 2}th cluster ...")
            row_indices = torch.arange(
                col_indices.shape[0] - (i + 1) * batch_size
            ).repeat_interleave(self.sparsity).to(col_indices.device)

            list_of_sparse_tensors.append(
                self.to_sparse_tensor(
                    row_indices=row_indices,
                    col_indices=col_indices[(i + 1) * batch_size:],
                    weight_slice=weights[(i + 1) * batch_size:]
                )
            )
        return list_of_sparse_tensors

    def to_sparse_tensor(
            self,
            row_indices: torch.Tensor,
            col_indices: torch.Tensor,
            weight_slice: torch.Tensor
    ) -> torch.sparse.FloatTensor:
        indices = torch.stack(
            (
                row_indices,
                col_indices.flatten()
            )
        )
        size = torch.Size(
            [weight_slice.shape[0], self.sample_dim]
        )
        return torch.sparse_coo_tensor(
            indices,
            weight_slice.flatten(),
            size
        )

    # verification function
    def forward_verifier(
            self, 
            dataset: TransformedDataset
    ) -> torch.Tensor:
        """
        Perform robust list learning of sparse linear classifiers for verification purpose.
        
        Parameters:
        dataset (torch.Tensor):       The input dataset.
                                    The first column is the label, which takes values in {0, 1}.
        
        Returns:
        weight_list (torch.Tensor): The list of weights for each combination.
        """

        # Extract features and labels
        # Assume the first column is the label column
        # and the rest are feature columns
        labels = dataset[:, 0] * 2 - 1
        labeled_features = labels.unsqueeze(1) * dataset[:, 1:]

        sample_size, self.sample_dim = labeled_features.shape[0], labeled_features.shape[1]
        
        # Generate row indices of all possible combinations of samples
        # dimension: [sample_size choose sparsity, sparsity]
        sample_indices = torch.combinations(torch.arange(sample_size), self.sparsity)

        # Generate column indices of all possible combinations of features
        # dimension: [sample_dim choose sparsity, sparsity]
        feature_indices = torch.combinations(torch.arange(self.sample_dim), self.sparsity)


        weight_list = torch.zeros(sample_indices.shape[0] * feature_indices.shape[0],  self.sample_dim)
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