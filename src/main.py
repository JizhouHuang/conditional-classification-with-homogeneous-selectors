import torch
import logging
from torch.utils.data import DataLoader
from .utils.helpers import UCIMedicalDataset, TransformedDataset
from .models.conditional_classification_finite_class import ConditionalLearnerForFiniteClass
from .models.robust_list_learning_of_sparse_linear_classifiers import RobustListLearner

def main():
    # Load the dataset
    file_path = "src/lib/hepatitis.data"  # Replace with your file path
    column_names = [
        "Class",  # Target (DIE=1, LIVE=2)
        "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
        "Anorexia", "Liver Big", "Liver Firm", "Spleen Palpable", "Spiders",
        "Ascites", "Varices", "Bilirubin", "Alk Phosphate", "SGOT",
        "Albumin", "Protime", "Histology"
    ]
    categorical_attr = ["Sex"]

    batch_size = 64
    num_iters = 1024
    train_ratio = 0.8
    num_classifier = 2048

    sparsity = 2
    margin = 0.0001

    print("reading HCI Medical dataset ...")
    uci_dataset = UCIMedicalDataset(file_path, column_names, categorical_attr)
    data_dim = uci_dataset.dim()
    init_weight = torch.zeros(data_dim, dtype=torch.float32)
    init_weight[0] = 1

    # Create training and testing datasets
    print("spliting dataset ...")
    train_dataset, test_dataset = torch.utils.data.random_split(
        uci_dataset, 
        [
            int(train_ratio * len(uci_dataset)), 
            len(uci_dataset) - int(train_ratio * len(uci_dataset))
        ]
    )
    print("converting training subset to tensor ...")
    train_dataset = next(
        iter(
            DataLoader(
                train_dataset, 
                batch_size=len(train_dataset)
            )
        )
    )
     # Learn the sparse classifiers
    print("initializing robust list learner ...")
    robust_list_learner = RobustListLearner(
        sparsity, 
        margin
    )
    print("running robust list learning algorithm ...")
    sparse_classifiers = robust_list_learner(train_dataset)

    print("creating training dataset ...")
    train_dataset = TransformedDataset(train_dataset)
    
    # Perform conditional learning
    print("initializing conditional learner ...")
    conditional_learner = ConditionalLearnerForFiniteClass(
        train_dataset, 
        num_classifier,
        num_iters, 
        lr_coeff=0.5, 
        train_ratio=train_ratio, 
        batch_size=batch_size
    )
    print("running conditional classification algorithm ...")
    selector_list = conditional_learner(sparse_classifiers)

    print("converting testing subset to tensor ...")
    test_dataset = next(
        iter(
            DataLoader(
                test_dataset, 
                batch_size=len(test_dataset)
            )
        )
    )
    print("creating testing dataset ...")
    test_dataset = TransformedDataset(test_dataset)
    labels, features = test_dataset[:]

    errors = (torch.sparse.mm(sparse_classifiers, features.T) >= 0) != labels
    selections = torch.matmul(selector_list, features.T) >= 0

    conditional_error_rate = (errors * selections).sum(dim=-1) / selections.sum(dim=-1)

    best_representation = torch.min(conditional_error_rate)

    # best_classifier, best_selector = sparse_classifiers[best_representation.indices].to_dense(), selector_list[best_representation.indices]
    
    print(f"final conditional classification error is {best_representation.values}")
    
if __name__ == "__main__":
    main()