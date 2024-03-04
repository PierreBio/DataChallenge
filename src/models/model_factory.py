import numpy as np

from src.models.gradient_boosting import GradientBoostingModel
from src.models.knn import KNNModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.naive_bayes import NaiveBayesModel
from src.models.neural_network import NeuralNetworkModel
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, X_train, Y_train, S_train, config, class_weights=None, sample_weights=None):
        if model_type == 'neural_network':
            config['weights'] = sample_weights
            return NeuralNetworkModel(X_train, Y_train, S_train, config)
        elif model_type == 'logistic_regression':
            return LogisticRegressionModel(X_train, Y_train, **config)
        elif model_type == 'svm':
            return SVMModel(X_train, Y_train, **config)
        elif model_type == 'naive_bayes':
            return NaiveBayesModel(X_train, Y_train, **config)
        elif model_type == 'gradient_boosting':
            return GradientBoostingModel(X_train, Y_train, **config)
        elif model_type == 'random_forest':
            class_weights_dict = {class_label: weight for class_label, weight in zip(np.unique(Y_train), class_weights)}
            config['class_weight'] = class_weights_dict
            return RandomForestModel(X_train, Y_train, **config)
        elif model_type == 'knn':
            return KNNModel(X_train, Y_train, **config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
