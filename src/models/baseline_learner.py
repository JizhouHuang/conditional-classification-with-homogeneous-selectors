import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from ..utils.simple_models import PredictiveModel

class LogisticRegLearner(PredictiveModel):
    def __init__(
            self,
            dataloader: DataLoader,
            device: torch.device = torch.device('cpu')
        ):

        # Initialize the logistic regression model
        model = self.train(
            model=LogisticRegression(),
            dataloader=dataloader
        )

        super().__init__(model, device)

class SVMLearner(PredictiveModel):
    def __init__(
            self, 
            dataloader: DataLoader,
            device: torch.device = torch.device('cpu')
    ):

        # Convert to numpy arrays and train model
        model = self.train(
            model=SVC(kernel='linear'),
            dataloader=dataloader
        )

        super().__init__(model, device)

class RandomForestLearner(PredictiveModel):
    def __init__(
            self, 
            dataloader: DataLoader,
            device: torch.device = torch.device('cpu')
    ):

        model = self.train(
            model=RandomForestClassifier(n_estimators=100),
            dataloader=dataloader
        )

        super().__init__(model, device)

class XGBoostLearner(PredictiveModel):
    def __init__(
            self, 
            dataloader: DataLoader,
            device: torch.device = torch.device('cpu')
    ):
        model = self.train(
            model=xgb.XGBClassifier(),
            dataloader=dataloader
        )

        super().__init__(model, device)
