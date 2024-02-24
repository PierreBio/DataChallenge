from src.models.logistic_regression import LogisticRegressionModel
from src.models.neural_network import NeuralNetworkModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, X_train, Y_train, config):
        if model_type == 'neural_network':
            return NeuralNetworkModel(X_train, Y_train, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
