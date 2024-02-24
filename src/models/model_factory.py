from src.models.logistic_regression import LogisticRegressionModel
from src.models.naive_bayes import NaiveBayesModel
from src.models.neural_network import NeuralNetworkModel
from src.models.svm import SVMModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, X_train, Y_train, config):
        if model_type == 'neural_network':
            return NeuralNetworkModel(X_train, Y_train, config)
        elif model_type == 'logistic_regression':
            return LogisticRegressionModel(X_train, Y_train, **config)
        elif model_type == 'svm':
            return SVMModel(X_train, Y_train, **config)
        elif model_type == 'naive_bayes':
            return NaiveBayesModel(X_train, Y_train, **config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
