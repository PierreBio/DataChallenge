from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

from src.models.base_model import BaseModel

class BestEpochCallback(Callback):
    def on_train_begin(self, logs=None):
        self.best_epoch = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch

class NeuralNetworkModel(BaseModel):
    def __init__(self, X_train, Y_train, config):
        super().__init__()
        self.model = self._build_model(config)
        self.best_epoch = 0
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
        best_epoch_callback = BestEpochCallback()
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)
        self.model.fit(X_train, Y_train_one_hot, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.1, callbacks=[best_epoch_callback, early_stopping, reduce_lr])
        self.best_epoch = best_epoch_callback.best_epoch

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions.argmax(axis=-1)