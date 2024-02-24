class BaseModel:
    def train(self, X_train, Y_train, **kwargs):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def evaluate(self, X_test, Y_test):
        raise NotImplementedError
