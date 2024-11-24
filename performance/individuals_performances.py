from models.model_1 import Model_1_gausian_nb
from models.model_2 import Model_2_forest
from models.model_3 import Model_3_SVM

if __name__ == "__main__":

    modules = [Model_1_gausian_nb, Model_2_forest, Model_3_SVM]

    for module in modules:

        model = module()
        print("The data that the model used")
        print(model.file)
        trained_model = model.get_model()
        model.performance(trained_model)
        metrics = model.confusion_matrix_and_metrics(trained_model)

        print("Performance Metrics:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"{metric.capitalize()}: {value}")

