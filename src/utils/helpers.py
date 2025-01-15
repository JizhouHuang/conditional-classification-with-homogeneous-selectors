from typing import Union, List, Tuple
import torch
import pandas as pd
from torch.utils.data import Dataset

def UCIMedicalDataset(
        file_path: str, 
        attributes: List[str], 
        categorical_attr: List[str], 
        label_name: str = "Class", 
        na_values: str = "?"
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

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # Convert to PyTorch tensors
    return torch.tensor(data.values, dtype=torch.float32)

def SliceDataWithRatio(
        ratio: float,
        data: torch.Tensor
) -> Tuple[torch.Tensor]:
    """
    Assume data is shuffled.

    Parameters:
    ratio (float): The ratio to split the data.
    data (torch.Tensor): The data to be split.

    Returns:
    Tuple[torch.Tensor]: A tuple containing two tensors, the first with the ratio of the data and the second with the remaining data.
    """
    cutoff_index = int(ratio * data.shape[0])
    return data[:cutoff_index], data[cutoff_index:]

class TransformedDataset(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            transform: callable = None
        ):
        """
        Initialize the dataset with a label mapping.

        Parameters:
        data (torch.Tensor):      The input data.
        transform (callable):   The label mapping function.
        """
        self.data = data
        self.transform = transform
        self.device = data.device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self, 
            idx: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        labels, features = sample[..., 0], sample[..., 1:]
        if self.transform:
            labels = self.transform(sample)
        return labels, features
    
    def dim(self) -> torch.Tensor:
        return self.data.size(1) - 1
    
    def set_transform(
            self, 
            transform: callable
        ) -> None:
        self.transform = transform

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
        self.classifiers = classifiers

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
            torch.sparse.mm(
                self.classifiers, 
                sample[1:].unsqueeze(-1)
            ).squeeze() >= 0
        ) != sample[0]
        # print(f"classifier shape: {self.classifiers.shape}")
        # print(f"labels shape: {labels.shape}")
        return labels.float()