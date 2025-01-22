from typing import List, Tuple, Any
import torch
import pandas as pd
from torch.utils.data import Dataset
from .simple_models import LinearModel

class UCIMedicalDataset:
    def __init__(
            self,
            file_path: str, 
            attributes: List[str], 
            label_name: str, 
            categorical_attr_names: List[str],
            binary_attr_names: List[str],
            sparse_attr_names: List[str], 
            label_true: int,
            label_false: int,
            attr_true: int,
            attr_false: int,
            na_values: str = "?",
            device: torch.device=torch.device('cpu')
        ) -> torch.Tensor:
        """
        Preprocess UCI Medical dataset by converting the data to PyTorch tensors.
        Dimension of the data: [num_samples, num_features+1] where the first column is the label.

        Parameters:
        file_path (str):                The path to the dataset file.
        attributes (list):              The list of attribute names.
        categorical_attr_names (list):  The list of categorical attribute names.
        label_name (str):               The name of the label column.
        na_values (str):                The string to recognize as missing values.
        """

        data = pd.read_csv(
            file_path, 
            header=None, 
            names=attributes, 
            na_values=na_values
        )

        # drop sparse attributes
        if sparse_attr_names:
            data = data.drop(
                labels=sparse_attr_names,
                axis=1
            )
        
        # move label column to the first
        data = data[
            [label_name] + [col for col in data.columns if col != label_name]
        ]

        # map binary attributes to {-1, +1}
        if binary_attr_names:
            data[binary_attr_names] = data[binary_attr_names].replace(
                {
                    attr_false: -1, 
                    attr_true: 1
                }
            )

        # Map labels to {0, 1}
        data[label_name] = data[label_name].replace({label_false: 0, label_true: 1})

        # Display basic info
        # print(data.head())
        # print(data.isnull().sum())  # Check for missing values

        # Convert categorical columns (e.g., "Sex") to one-hot encoding if needed
        if categorical_attr_names:
            data = pd.get_dummies(
                data, 
                columns=categorical_attr_names
            )

        # Fill missing values (e.g., with the column mean)
        data.fillna(
            data.mean(), 
            inplace=True
        )

        # Center and normalize the features (excluding the label column)
        col_to_normalize = [col for col in data.columns if col != label_name]
        data[col_to_normalize] = (data[col_to_normalize] - data[col_to_normalize].mean()) / data[col_to_normalize].std()

        self.device = device

        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        # store as tensor
        self.data = torch.tensor(data.values, dtype=torch.float32).to(device)

    def slice_with_ratio(
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

class TransformedDataset(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            predictor: Any = None
        ):
        """
        Initialize the dataset with a label mapping.
        Note that the label map will generate cluster_size {0, 1} labels for each example.
        A label of "1" indicates the corresponding sparse classifier disagrees with the 
        true original data label.

        Parameters:
        data (torch.Tensor):    The input data.
        predictor:              A tuple of classifier(s) and its corresponding prediction method.
                                Classifier(s) can be any class, while the prediction method must
                                take the classifier(s) and the feature in the form of torch.Tensor as
                                inputs, then outputs a label in the form of torch.Tensor.
        """
        self.data = data
        self.trans_labels = None
        self.predictor = predictor
        self.label_map()
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
    
    def label_map(self) -> None:
        """
        Map the labels according to the given predictor.
        """
        if self.predictor and hasattr(self.predictor, 'errors'):
            self.trans_labels = self.predictor.errors(
                X=self.data[:, 1:],     # [data_batch_size, dim_sample]
                y=self.data[:, 0]       # [data_batch_size]
            ).t()                       # [data_batch_size, cluster_size]
        else:
            self.trans_labels = self.data[:, 0]
    
    def set_predictor(
            self,
            predictor: Any
    ):
        self.predictor = predictor
        self.label_map()

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
    