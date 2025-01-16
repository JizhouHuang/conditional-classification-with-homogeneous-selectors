import torch
from .experiments.experiment import Experiment
from .utils.helpers import UCIMedicalDataset

def main():

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Experiment configuration
    experiment_id = 1
    config_file_path = 'src/config/models.yaml'

    # Initialize the experiment
    experiment = Experiment(
        experiment_id=experiment_id, 
        config_file_path=config_file_path
    ).to(device)

    # Load the dataset
    file_path = "src/datasets/hepatitis.data"  # Replace with your file path
    column_names = [
        "Class",  # Target (DIE=1, LIVE=2)
        "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
        "Anorexia", "Liver Big", "Liver Firm", "Spleen Palpable", "Spiders",
        "Ascites", "Varices", "Bilirubin", "Alk Phosphate", "SGOT",
        "Albumin", "Protime", "Histology"
    ]
    categorical_attr = ["Sex"]

    # Load and preprocess the data
    uci_data = UCIMedicalDataset(
        file_path=file_path,
        attributes=column_names,
        categorical_attr=categorical_attr,
        device=device
    )
    train_data, test_data = uci_data.SliceDataWithRatio(0.8)

    # Run the experiment
    experiment(train_data, test_data)

if __name__ == "__main__":
    main()