from typing import Union, List, Tuple
import torch
import pandas as pd
from torch.utils.data import Dataset

class UCIMedicalDataset:
    def __init__(
            self,
            file_path: str, 
            attributes: List[str], 
            categorical_attr: List[str], 
            label_name: str = "Class", 
            na_values: str = "?",
            device: torch.device=torch.device('cpu')
        ) -> torch.Tensor:
        """
        Preprocess UCI Medical dataset by converting the data to PyTorch tensors.
        Dimension of the data: [num_samples, num_features+1] where the first column is the label.

        Parameters:
        file_path (str):            The path to the dataset file.
        attributes (list):          The list of attribute names.
        categorical_attr (list):    The list of categorical attribute names.
        label_name (str):           The name of the label column.
        na_values (str):            The string to recognize as missing values.
        """

        data = pd.read_csv(
            file_path, 
            header=None, 
            names=attributes, 
            na_values=na_values
        )

        # Display basic info
        # print(data.head())
        # print(data.isnull().sum())  # Check for missing values

        # Fill missing values (e.g., with the column mean)
        data.fillna(
            data.mean(), 
            inplace=True
        )

        # Convert categorical columns (e.g., "Sex") to one-hot encoding if needed
        data = pd.get_dummies(
            data, 
            columns=categorical_attr
        )

        # Map labels to {0, 1}
        data[label_name] = data[label_name].map({1: 0, 2: 1})

        # Center and normalize the features (excluding the label column)
        col_to_normalize = [col for col in data.columns if col != label_name]
        data[col_to_normalize] = (data[col_to_normalize] - data[col_to_normalize].mean()) / data[col_to_normalize].std()

        self.device = device

        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        # store as tensor
        self.data = torch.tensor(data.values, dtype=torch.float32).to(device)

    def SliceDataWithRatio(
            self,
            ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assume data is shuffled.

        Parameters:
        ratio (float): The ratio to split the data.

        Returns:
        Tuple[torch.Tensor]: A tuple containing two tensors, the first with the ratio of the data and the second with the remaining data.
        """
        cutoff_index = int(ratio * self.data.shape[0])
        return self.data[:cutoff_index], self.data[cutoff_index:]

def Classify(
        classifier: torch.Tensor,
        data: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(classifier, data) > 0

def SparseClassify(
        classifier: torch.sparse.FloatTensor,
        data: torch.Tensor
) -> torch.Tensor:
    return torch.sparse.mm(classifier, data) > 0

class TransformedDataset(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            transform: torch.sparse.FloatTensor = None
        ):
        """
        Initialize the dataset with a label mapping.
        Note that the transform will generate cluster_size {0, 1} labels for each example.
        A label of "1" indicates the corresponding sparse classifier disagrees with the 
        true original data label.

        Parameters:
        data (torch.Tensor):      The input data.
        transform (callable):   The label mapping function.
        """
        self.data = data
        self.trans_labels = None
        self.set_transform(transform=transform)
        self.device = data.device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self, 
            idx: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trans_labels[idx], self.data[idx, 1:]
    
    def dim(self) -> torch.Tensor:
        return self.data.size(1) - 1
    
    def set_transform(
            self, 
            transform: torch.sparse.FloatTensor
        ) -> None:
        if isinstance(transform, torch.Tensor) and transform.is_sparse:
            features = self.data[:, 1:]
            self.trans_labels = (
                SparseClassify(
                    classifier=transform,
                    data=features.T
                ).T # [num_sample, num_classifier]
            ) != self.data[:, 0].unsqueeze(-1)
        else:
            self.trans_labels = self.data[:, 0]

class FixedIterationLoader:
    def __init__(self, dataloader, max_iterations):
        self.dataloader = dataloader
        self.max_iterations = max_iterations

    def __iter__(self):
        self.data_iter = iter(self.dataloader)
        self.iter_count = 0
        return self

    def __next__(self):
        if self.iter_count >= self.max_iterations:
            raise StopIteration
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        self.iter_count += 1
        return batch
    
# Deprecated
class LabelMapping:
    """
    Map the data label according to a given sparse halfspace.
    """ 
    def __init__(
            self, 
            classifiers: torch.sparse.FloatTensor
        ):
        """
        Initialize the label mapping.

        Parameters:
        classifier (torch.Tensor):  Sparse classifiers of sparse representation.
                                    [classifier_batch_size, dim_sample]
        """
        self.classifiers = classifiers.to_dense()

    def __call__(
            self, 
            sample: torch.Tensor
        ) -> torch.Tensor:
        """
        Change y to y != classifier(x) for the given sample.

        Parameters:
        sample (torch.Tensor):  The input sample.
                                [num_features+1] where the first column is a {0, 1}-valued label.
        
        Returns:
        torch.Tensor:           The transformed labels.
                                [classifier_batch_size]
        """

        labels = (
            torch.mv(
                self.classifiers, 
                sample[1:]
            ) >= 0
        ) != sample[0]
        # print(f"classifier shape: {self.classifiers.shape}")
        # print(f"labels shape: {labels.shape}")
        return labels.float()