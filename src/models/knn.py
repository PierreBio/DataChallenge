from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.models.base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, X_train, Y_train, **params):
        super().__init__()
        self.model = KNeighborsClassifier(**params)
        self.train(X_train, Y_train)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        return accuracy_score(Y_test, predictions)
