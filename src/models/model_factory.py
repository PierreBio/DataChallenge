from src.models.logistic_regression import LogisticRegressionModel
from src.models.neural_network import NeuralNetworkModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, X_train, Y_train, **params):
        if model_type == 'logistic_regression':
            return LogisticRegressionModel(X_train, Y_train, **params)
        elif model_type == 'neural_network':
            return NeuralNetworkModel(X_train, Y_train, **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
