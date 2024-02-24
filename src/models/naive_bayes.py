from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from src.models.base_model import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self, X_train, Y_train, **params):
        super().__init__()
        model_type = params.pop('type', 'GaussianNB')

        if model_type == 'GaussianNB':
            self.model = GaussianNB(**params)
        elif model_type == 'MultinomialNB':
            self.model = MultinomialNB(**params)
        elif model_type == 'BernoulliNB':
            self.model = BernoulliNB(**params)
        else:
            raise ValueError(f"Unsupported Naive Bayes model type: {model_type}")

        self.train(X_train, Y_train)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        return accuracy_score(Y_test, predictions)