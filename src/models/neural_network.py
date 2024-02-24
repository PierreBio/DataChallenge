from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from src.models.base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self, X_train, Y_train, config):
        super().__init__()
        self.model = self._build_model(config)
        self.train(X_train, Y_train, config)

    def _build_model(self, config):
        model = Sequential()
        model.add(Dense(config['layers'][0]['units'], activation=config['layers'][0]['activation'], input_shape=(config['input_shape'],)))

        for layer in config['layers'][1:]:
            model.add(Dense(layer['units'], activation=layer['activation']))
            if 'dropout' in layer:
                model.add(Dropout(layer['dropout']))

        model.add(Dense(config['output_units'], activation=config['output_activation']))

        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer, loss=config['loss'], metrics=config['metrics'])
        return model

    def train(self, X_train, Y_train, config):
        Y_train_one_hot = to_categorical(Y_train, num_classes=config['output_units'])
        self.model.fit(X_train, Y_train_one_hot, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.1)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions.argmax(axis=-1)

    def evaluate(self, X_test, Y_test):
        Y_test_one_hot = to_categorical(Y_test)
        return self.model.evaluate(X_test, Y_test_one_hot)
