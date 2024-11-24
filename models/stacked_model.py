import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.preprocess import Preprocess

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier


class Stacked_Models():

    def __init__(self, path_1, path_2, path_3):


        if os.path.exists(path_1):
            with open(path_1, "rb") as file:
                self.model_1 = pickle.load(file)
            print("Model 1 loaded successfully!")

        if os.path.exists(path_2):
            with open(path_2, "rb") as file:
                self.model_2 = pickle.load(file)
            print("Model 2 loaded successfully!")

        if os.path.exists(path_3):
            with open(path_3, "rb") as file:
                self.model_3 = pickle.load(file)
            print("Model 3 loaded successfully!")

        route: str = "../data/Loan.csv"
        data = Preprocess(route)

        self.file: pd.DataFrame = data.opened_file
        self.df_dummies = data.get_dummies(self.file)

        self.X = self.df_dummies.drop(columns="LoanApproved")

        self.y = self.df_dummies["LoanApproved"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def get_model(self, meta_model=None):

        """
        Builds, trains, and saves a stacking model using three base models and a meta-model. If a saved model exists,
        it loads the model instead.

        Parameters
        ----------
        meta_model : estimator, optional
            The meta-model to use as the final estimator in the stacking classifier. If not provided,
            a RandomForestClassifier is used by default.

        Raises
        ------
        ValueError
            If the three base models (`model_1`, `model_2`, `model_3`) are not already defined.

        Returns
        -------
        StackingClassifier
            The trained stacking model, either loaded from a saved file or newly trained.
        """

        save_path = "../saved_models/saved4/stacked_model.pkl"

        if os.path.exists(save_path):
            with open(save_path, "rb") as file:
                model = pickle.load(file)
            print("Model loaded successfully!")

        else:


            if not hasattr(self, "model_1") or not hasattr(self, "model_2") or not hasattr(self, "model_3"):
                raise ValueError("All three base models must be loaded before building the stacking model.")

            if meta_model is None:
                meta_model = RandomForestClassifier()

            stacking_model = StackingClassifier(
                estimators=[
                    ("gaussian_nb", self.model_1),  # Gaussian Naive Bayes
                    ("random_forest", self.model_2),  # Random Forest
                    ("svm", self.model_3),  # SVM
                ],
                final_estimator=meta_model,
                cv=5
            )


            stacking_model.fit(self.X_train, self.y_train)
            print("Stacking model trained successfully!")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as file:
                pickle.dump(stacking_model, file)
            print(f"Model saved successfully at {save_path}!")

            return stacking_model

        return model

    def performance(self, model=None):

        """
        Evaluates the performance of a model by plotting the ROC curve, calculating the AUC score, and displaying it.

        Parameters
        ----------
        model : optional
            The trained model to evaluate. If None, the default model is retrieved using `self.get_model()`.

        Raises
        ------
        ValueError
            If the model does not have `predict_proba` or `decision_function` methods.

        Returns
        -------
        float
            The AUC (Area Under the Curve) score of the model's performance.
        """

        if model is None:
            model = self.get_model()


        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(self.X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_probs = model.decision_function(self.X_test)
        else:
            raise ValueError("Model does not have `predict_proba` or `decision_function`.")


        fpr, tpr, _ = roc_curve(self.y_test, y_probs)
        roc_auc = auc(fpr, tpr)

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

        print(f"AUC Score: {roc_auc:.2f}")

        return roc_auc

    def confusion_matrix_and_metrics(self, model):

        """
        Calculates and displays the confusion matrix and various performance metrics for a given model.

        Parameters
        ----------
        model : estimator
            The trained model to evaluate.

        Returns
        -------
        dict
            A dictionary containing the following metrics:
            - "accuracy": The accuracy of the model.
            - "precision": The precision of the model.
            - "recall": The recall of the model.
            - "f1_score": The F1 score of the model.
            - "confusion_matrix": The confusion matrix.

        Displays
        --------
        Confusion Matrix
            A visual representation of the confusion matrix using matplotlib.
        """


        y_pred_train = model.predict(self.X_train)
        cm_train = confusion_matrix(self.y_train, y_pred_train)

        accuracy_train = accuracy_score(self.y_train, y_pred_train)
        precision_train = precision_score(self.y_train, y_pred_train, average="binary")
        recall_train = recall_score(self.y_train, y_pred_train, average="binary")
        f1_train = f1_score(self.y_train, y_pred_train, average="binary")

        y_pred_test = model.predict(self.X_test)
        cm_test = confusion_matrix(self.y_test, y_pred_test)

        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        precision_test = precision_score(self.y_test, y_pred_test, average="binary")
        recall_test = recall_score(self.y_test, y_pred_test, average="binary")
        f1_test = f1_score(self.y_test, y_pred_test, average="binary")


        print("Confusion Matrix - Train:")
        ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greens")
        plt.title("Confusion Matrix - Train")
        plt.show()

        # Display Confusion Matrices
        print("Confusion Matrix - Test:")
        ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap="Blues")
        plt.title("Confusion Matrix - Test")
        plt.show()


        return {
            "train_metrics": {
                "accuracy": accuracy_train,
                "precision": precision_train,
                "recall": recall_train,
                "f1_score": f1_train,
                "confusion_matrix": cm_train,
            },
            "test_metrics": {
                "accuracy": accuracy_test,
                "precision": precision_test,
                "recall": recall_test,
                "f1_score": f1_test,
                "confusion_matrix": cm_test,
            },
        }

    def get_prediction(self, data):

        """
        Generates predictions for the given input data using the trained model.

        Parameters
        ----------
        data : array-like or pd.DataFrame
            The input data for which predictions are to be generated.

        Returns
        -------
        array
            Predicted labels for the input data.
        """

        model = self.get_model()
        prediction = model.predict(data)

        return prediction


if __name__ == "__main__":

    model_path_1 = "../saved_models/saved1/gaussian_nb_best_model.pkl"
    model_path_2 = "../saved_models/saved2/forest_best_model.pkl"
    model_path_3 = "../saved_models/saved3/svm_model.pkl"


    model = Stacked_Models(path_1=model_path_1, path_2=model_path_2, path_3=model_path_3)

    print(model.file)
    trained_model = model.get_model()

    model.performance(trained_model)
    metrics = model.confusion_matrix_and_metrics(trained_model)

    print("Performance Metrics:")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric.capitalize()}: {value}")
