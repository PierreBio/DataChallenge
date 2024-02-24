from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, X_train, Y_train, **params):
        super().__init__()
        self.model = LogisticRegression(**params)
        self.train(X_train, Y_train)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, Y_test):
        return self.model.score(X_test, Y_test)