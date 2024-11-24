from models.stacked_model import Stacked_Models


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