from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel

class SVMModel(BaseModel):
    def __init__(self, X_train, Y_train, **params):
        super().__init__()
        self.model = make_pipeline(StandardScaler(), SVC(**params))
        self.train(X_train, Y_train)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        return accuracy_score(Y_test, predictions)
