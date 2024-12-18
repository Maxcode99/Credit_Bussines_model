import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.preprocess import Preprocess
from models.structure import Structure

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV




class Model_2_forest(Structure):
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

    def get_model(self, save_path="../saved_models/saved2/forest_best_model.pkl"):

        """
        Retrieves or trains a Random Forest model and saves/loads it as needed.

        Parameters
        ----------
        save_path : str, optional
            The file path where the model will be saved or loaded from.
            Default is "../saved_models/saved2/forest_best_model.pkl".

        Returns
        -------
        Union[RandomForestClassifier, Tuple[float, float, RandomForestClassifier]]
            If the model already exists at the specified path, the saved model is loaded and returned.
            If the model does not exist:
                - Trains a Random Forest model with cross-validation.
                - Saves the trained model to the specified path.
                - Returns a tuple containing:
                  - The mean training accuracy score (float).
                  - The mean testing accuracy score (float).
                  - The trained Random Forest model.
        """

        model_path = "../saved_models/saved2/forest_best_model.pkl"

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("Model loaded successfully!")
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                n_jobs=-1
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            cv_results = cross_validate(
                model,
                self.X,
                self.y,
                cv=cv,
                scoring="accuracy",
                return_train_score=True
            )

            train_scores = np.mean(cv_results["train_score"])
            test_scores = np.mean(cv_results["test_score"])


            model.fit(self.X, self.y)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(model_path, "wb") as file:
                pickle.dump(model, file)
            print(f"Model saved successfully at {model_path}!")

            return train_scores, test_scores, model

        return model

    def get_hyperparameter(self, save_path="../saved_models/saved2/forest_best_model.pkl"):

        """
        Tunes the hyperparameters of a Random Forest model using RandomizedSearchCV,
        saves the best model, and returns it.

        Parameters
        ----------
        save_path : str, optional
            The file path where the best model will be saved.
            Default is "../saved_models/saved2/forest_best_model.pkl".

        Returns
        -------
        RandomForestClassifier
            The Random Forest model with the best hyperparameters obtained from RandomizedSearchCV.
        """

        param_distributions = {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'max_depth': [1, 2, 3, 4],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        }

        rf = RandomForestClassifier(random_state=42)


        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=50,
            scoring='accuracy',
            cv=5,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(self.X_train, self.y_train)

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
        Evaluates the performance of a model by plotting the ROC curve and calculating the AUC.

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
        None
            Displays the ROC curve with the AUC score.
        """

        if model is None:
            model = self.get_model()

        # Get predicted probabilities
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
        plt.title("Receiver Operating Characteristic (ROC) Curve (Random Forest)")
        plt.legend(loc="lower right")
        plt.show()

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


        y_pred_test = model.predict(self.X_test)
        cm_test = confusion_matrix(self.y_test, y_pred_test)


        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        precision_test = precision_score(self.y_test, y_pred_test, average="binary")
        recall_test = recall_score(self.y_test, y_pred_test, average="binary")
        f1_test = f1_score(self.y_test, y_pred_test, average="binary")


        y_pred_train = model.predict(self.X_train)
        cm_train = confusion_matrix(self.y_train, y_pred_train)


        accuracy_train = accuracy_score(self.y_train, y_pred_train)
        precision_train = precision_score(self.y_train, y_pred_train, average="binary")
        recall_train = recall_score(self.y_train, y_pred_train, average="binary")
        f1_train = f1_score(self.y_train, y_pred_train, average="binary")

        print("Confusion Matrix - Train:")
        ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greens")
        plt.title("Confusion Matrix - Train")
        plt.show()

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
    model = Model_2_forest()
    trained_model = model.get_model()

    model.performance(trained_model)
    metrics = model.confusion_matrix_and_metrics(trained_model)

    print("Performance Metrics:")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric.capitalize()}: {value}")


    # best = model.get_hyperparameter()
    # print(best)