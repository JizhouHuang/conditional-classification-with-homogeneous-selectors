import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class UCIMedicalDataset(Dataset):
    def __init__(self, file_path, attributes, categorical_attr, label_name="Class", na_values="?"):
        """
        Initialize the UCI Medical dataset by converting the data to PyTorch tensors.
        Dimension of the data: [num_samples, num_features+1] where the first column is the label.

        Parameters:
        file_path (str):            The path to the dataset file.
        attributes (list):          The list of attribute names.
        categorical_attr (list):    The list of categorical attribute names.
        label_name (str):           The name of the label column.
        binary_attr (list):         The list of binary attribute names.
        na_values (str):            The string to recognize as missing values.
        transform (callable):       Optional transform to be applied on a sample.
        """

        data = pd.read_csv(file_path, header=None, names=attributes, na_values=na_values)

        # Display basic info
        # print(data.head())
        # print(data.isnull().sum())  # Check for missing values

        # Fill missing values (e.g., with the column mean)
        data.fillna(data.mean(), inplace=True)

        # Convert categorical columns (e.g., "Sex") to one-hot encoding if needed
        data = pd.get_dummies(data, columns=categorical_attr)

        # Map labels to {0, 1}
        data[label_name] = data[label_name].map({1: 0, 2: 1})

        # Center and normalize the features (excluding the label column)
        col_to_normalize = [col for col in data.columns if col != label_name]
        data[col_to_normalize] = (data[col_to_normalize] - data[col_to_normalize].mean()) / data[col_to_normalize].std()

        # Convert to PyTorch tensors
        self.data = torch.tensor(data.values, dtype=torch.float32)

        # self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    def dim(self):
        return self.data.size(1) - 1

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Initialize the dataset with a label mapping.

        Parameters:
        dataset (Dataset):      The input dataset.
        transform (callable):   The label mapping function.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        labels, features = sample[0], sample[1:]
        if self.transform:
            labels = self.transform(sample)
        return labels, features
    
    def dim(self):
        return self.dataset.size(1) - 1
    
    def set_transform(self, transform):
        self.transform = transform

class LabelMapping:
    """
    Map the data label according to a given sparse halfspace.
    """ 
    def __init__(self, classifiers):
        """
        Initialize the label mapping.

        Parameters:
        classifier (torch.Tensor):  Sparse classifiers of sparse representation.
                                    [classifier_batch_size, dim_sample]
        """
        self.classifiers = classifiers

    def __call__(self, sample):
        """
        Change y to y != classifier(x) for the given sample.

        Parameters:
        sample (torch.Tensor):  The input sample.
                                [num_features+1] where the first column is a {0, 1}-valued label.
        
        Returns:
        torch.Tensor:           The transformed labels.
                                [classifier_batch_size]
        """
        labels = (torch.matmul(self.classifiers, sample[1:]) >= 0) != sample[0]
        # print(f"classifier shape: {self.classifiers.shape}")
        # print(f"labels shape: {labels.shape}")
        return labels.float()