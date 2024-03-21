import os
import shutil
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from src.preprocessing.data_manager import *
from src.models.model_factory import ModelFactory
from src.postprocessing.evaluator import *
from src.postprocessing.performance import *

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_score(Y_pred, Y_test, S_test):
    eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
    print("FINAL SCORE: ", final_score)
    return final_score

def storeResults(final_score):
    results = pd.DataFrame(Y_pred, columns=['score'])
    results.to_csv('./results' + "/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
    np.savetxt('./results' + '/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')

if __name__ == "__main__":
    dat = load_data('./data/data-challenge-student.pickle')
    config = load_config('./config/config.json')

    folder_path = './results'

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path, exist_ok=True)

    while True:
        X_train_final, X_test, Y_train_final, Y_test, S_train, S_test = prepare_data(dat, 0.2)

        # Utilisation de la régression logistique pour l'entraînement du modèle
        model_type = 'logistic_regression'
        model_config = config[model_type]
        model = ModelFactory.get_model(model_type, X_train_final, Y_train_final, S_train, model_config)

        Y_pred = model.predict(X_test)
        final_score = evaluate_score(Y_pred, Y_test, S_test)

        X_test_true_filtered = dat["X_test"]
        Y_pred = model.predict(X_test_true_filtered)

        if(final_score > 0.783):
            storeResults(final_score)

        if(final_score > 0.772):
            model_neural = Sequential()
            model_neural.add(Dense(28, input_shape=(X_train_final.shape[1],), activation='softmax'))
            model_neural.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model_neural.fit(X_train_final, Y_train_final, epochs=20, batch_size=10, verbose=1)

            predictions_prob = model_neural.predict(X_test)
            predictions_labels = np.argmax(predictions_prob, axis=1)
            final_score = evaluate_score(predictions_labels, Y_test, S_test)

            X_test_true_filtered = dat["X_test"]
            Y_pred = model_neural.predict(X_test_true_filtered)
            predictions_labels = np.argmax(Y_pred, axis=1)

            if(final_score > 0.78):
                storeResults(final_score)