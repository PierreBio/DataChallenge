from tensorflow.keras.utils import to_categorical

def train_model(model, X_train, Y_train, model_type='logistic_regression', epochs=10, batch_size=32):
    if model_type == 'logistic_regression':
        model.fit(X_train, Y_train)
    elif model_type == 'neural_network':
        Y_train_one_hot = to_categorical(Y_train)
        model.fit(X_train, Y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_split=0.1)
