from abc import ABC, abstractmethod
from preprocessing.preprocess import Preprocess
import pandas as pd
from sklearn.model_selection import train_test_split

class Structure(ABC):

    @abstractmethod
    def __init__(self):
        route: str = "../data/Loan.csv"
        data = Preprocess(route)
        self.file: pd.DataFrame = data.opened_file
        self.df_dummies = data.get_dummies(self.file)
        self.X = self.df_dummies.drop(columns="LoanApproved")
        self.y = self.df_dummies["LoanApproved"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    @abstractmethod
    def get_model(self, save_path="../saved_models/saved2/gaussian_nb_best_model.pkl"):
        ...

    @abstractmethod
    def get_hyperparameter(self, save_path="../saved_models/saved2/gaussian_nb_best_model.pkl"):
        ...

    @abstractmethod
    def performance(self, model=None):
        ...

    @abstractmethod
    def confusion_matrix_and_metrics(self, model):
        ...

