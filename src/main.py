import torch
import logging
from torch.utils.data import DataLoader, Subset
from .utils.helpers import UCIMedicalDataset, SliceDataWithRatio, TransformedDataset
from .models.conditional_classification_finite_class import ConditionalLearnerForFiniteClass
from .models.robust_list_learning_of_sparse_linear_classifiers import RobustListLearner

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    num_iter = 2048
    train_ratio = 0.7
    robust_learning_ratio = 0.7
    num_classifier_cluster = 32

    sparsity = 3
    margin = 0.0001

    print("main - preparing HCI Medical dataset ...")
    print("main - dividing HCI dataset into training and testing sets ...")
    train_dataset, test_dataset = SliceDataWithRatio(
        ratio=train_ratio,
        data=UCIMedicalDataset(file_path, column_names, categorical_attr).to(device)
    )
    
    # Learn the sparse classifiers
    print("main - initializing robust list learner ...")
    robust_list_learner = RobustListLearner(
        sparsity=sparsity, 
        margin=margin,
        num_slices=num_classifier_cluster
    ).to(device)
    rl_dataloader = DataLoader(
        TransformedDataset(train_dataset),
        batch_size=int(robust_learning_ratio * train_dataset.shape[0]),
        shuffle=True
    )
    print("main - running robust list learning algorithm ...")
    sparse_classifier_clusters = robust_list_learner(
        next(iter(rl_dataloader))
    )

    # Perform conditional learning
    print("main - initializing conditional learner ...")
    conditional_learner = ConditionalLearnerForFiniteClass(
        dim_sample = train_dataset.shape[1] - 1,
        num_iter=num_iter, 
        lr_coeff=0.5, 
        train_ratio=train_ratio, 
        batch_size=batch_size
    ).to(device)
    print("main - running conditional classification algorithm ...")
    classifier, selector = conditional_learner(
        data=train_dataset,
        sparse_classifier_clusters=sparse_classifier_clusters
    )
    
    print(f"main - best classifier is: {classifier}")
    print(f"main - best selector is: {selector}")

def conditional_classification_error(
        test_dataset: Subset,
        sparse_classifier_clusters: list[torch.sparse.FloatTensor],
        selector_list: torch.Tensor
) -> torch.Tensor:
    print("main - converting testing subset to tensor ...")
    test_dataset = next(
        iter(
            DataLoader(
                test_dataset, 
                batch_size=len(test_dataset)
            )
        )
    )
    print("main - creating testing dataset ...")
    test_dataset = TransformedDataset(test_dataset)
    labels, features = test_dataset[:]
    print(f"main - testing labels shape: {labels.shape}")
    print(f"main - testing features shape: {features.shape}")

    errors = torch.cat(
        [
            (
                torch.sparse.mm(classifiers, features.T) >= 0
            ) != labels for classifiers in sparse_classifier_clusters
        ],
        dim=0
    )
    selections = torch.matmul(selector_list, features.T) >= 0

    conditional_error_rate = (errors * selections).sum(dim=-1) / selections.sum(dim=-1)
    # replace NaN to 1
    conditional_error_rate[torch.isnan(conditional_error_rate)] = 1

    return torch.min(conditional_error_rate, dim=0)

    
if __name__ == "__main__":
    main()