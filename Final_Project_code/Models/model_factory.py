from abc import ABC
from typing import Any
from Models.poisson_model import PoissonModel
from Models.random_forest_model import RandomForestModel
from Models.svm_model import SVmModel
from Models.xgboost_model import XGBoost
from Models.logistic_regression import LogisticRegress


class ModelFactory(ABC):
    """
    """

    def __init__(self):
        self._models = {"Poisson": PoissonModel(),
                        "RandomForest": RandomForestModel(),
                        "SVM": SVmModel(),
                        "XGBoost": XGBoost(),
                        "LogisticRegression": LogisticRegress(),
                        "AnotherModel": "AnotherModel()"}

    def get_model(self, model_class: str) -> Any:
        try:
            return self._models[model_class]
        except:
            raise Exception(f"No Model Name for {model_class}")
