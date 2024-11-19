import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.preprocess import Preprocess

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV




class Model_1():
    def __init__(self):
        route: str = "../data/Loan.csv"
        data = Preprocess(route)
        self.file: pd.DataFrame = data.opened_file
        self.df_dummies = data.get_dummies()
        self.X = self.df_dummies.drop(columns="LoanApproved")
        self.y = self.df_dummies["LoanApproved"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def get_model(self, save_path="../saved_models/saved2/gaussian_nb_best_model.pkl"):

        model_path = "../saved_models/saved2/gaussian_nb_best_model.pkl"

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("Model loaded successfully!")
        else:
            model = GaussianNB()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            cv_results = cross_validate(
                model, self.X, self.y, cv=cv, scoring="accuracy", return_train_score=True
            )

            train_scores = np.mean(cv_results["train_score"])
            test_scores = np.mean(cv_results["test_score"])

            model.fit(self.X, self.y)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(model_path, "wb") as file:
                pickle.dump(model, file)
            print("Model saved successfully!")

            return train_scores, test_scores, model

        return model

    def get_hyperparameter(self, save_path="../saved_models/saved2/gaussian_nb_best_model.pkl"):
        param_distributions = {
            "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }

        model = GaussianNB()

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=10,
            scoring="accuracy",
            cv=5,
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(self.X, self.y)

        best_model = random_search.best_estimator_
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as file:
            pickle.dump(best_model, file)
        print(f"Model saved successfully at {save_path}!")

        return best_model

    def performance(self, model=None):
        """
        Plots the AUC-ROC curve for a given model and the test data.

        Parameters:
        - model: The trained model (optional, default is None).
        """
        # If no model is passed, load the default model
        if model is None:
            model = self.get_model()

        # Get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(self.X_test)[:, 1]  # Get probabilities for the positive class
        elif hasattr(model, "decision_function"):
            y_probs = model.decision_function(self.X_test)  # Get decision function scores
        else:
            raise ValueError("Model does not have `predict_proba` or `decision_function`.")

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)

        # Calculate AUC score
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()

    def confusion_matrix(self):
        ...


    def get_prediction(self, data):

        model = self.get_model()
        prediction = model.predict(data)

        return prediction


if __name__ == "__main__":
    model = Model_1()
    print(model.file)
    trained_model = model.get_model()
    # best = model.get_hyperparameter()
    # print(best)
    model.performance(trained_model)
