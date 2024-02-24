from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from src.models.base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self, X_train, Y_train, input_shape, num_classes, **model_config):
        super().__init__()
        self.model = self._build_model(input_shape, num_classes, **model_config)
        self.train(X_train, Y_train, model_config.get('epochs', 10), model_config.get('batch_size', 32))

    def _build_model(self, input_shape, num_classes, **config):
        first_layer_units = config.get('first_layer_units', 512)
        model = Sequential([
            Dense(first_layer_units, activation='relu', input_shape=(input_shape,)),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=config.get('optimizer', 'adam'),
                      loss=config.get('loss', 'categorical_crossentropy'),
                      metrics=config.get('metrics', ['accuracy']))
        return model

    def train(self, X_train, Y_train, epochs=10, batch_size=32):
        Y_train_one_hot = to_categorical(Y_train)
        self.model.fit(X_train, Y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions.argmax(axis=-1)

    def evaluate(self, X_test, Y_test):
        Y_test_one_hot = to_categorical(Y_test)
        return self.model.evaluate(X_test, Y_test_one_hot)
