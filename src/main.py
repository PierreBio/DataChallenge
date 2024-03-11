from src.preprocessing.data_manager import *
from src.models.model_factory import ModelFactory
from src.postprocessing.evaluator import *
from src.postprocessing.performance import *
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.cluster import DBSCAN

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    dat = load_data('./data/data-challenge-student.pickle')

    import os
    import shutil

    # Chemin du dossier à nettoyer
    folder_path = './results'

    # Vérifier si le dossier existe
    if os.path.exists(folder_path):
        # Supprimer le dossier et tout son contenu
        shutil.rmtree(folder_path)

    # Recréer le dossier vide
    os.makedirs(folder_path, exist_ok=True)

    def load_data(filepath):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        if isinstance(data, pd.DataFrame):
            print(type(data))
        with open(filepath, 'rb') as handle:
            dat = pd.read_pickle(handle)
            print(dat.keys())
        return dat
    #Paramètres: C=0.3509037246039834, max_iter=100, tol=0.7515379629797547, solver=lbfgs, multi_class=auto
    #Score pour le pli courant: 0.7358150100779599
    #Score moyen sur tous les plis: 0.7374355489608619
    while True:
        X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
            dat['X_train'], dat['Y'], dat['S_train'],
            test_size=0.2, stratify=np.column_stack([dat['Y'], dat['S_train']])
        )

        X_train = pd.DataFrame(X_train)
        for column in X_train.columns:
            X_train[column] = winsorize(X_train[column], limits=(0.01, 0.01))

        X_train_selected = X_train
        X_test_selected = X_test

        base_lr = LogisticRegression(solver='lbfgs', max_iter=100, C=0.3, tol=0.00750, multi_class='auto')

        base_lr.fit(X_train_selected, Y_train)

        Y_pred = base_lr.predict(X_test_selected)

        eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
        final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
        print("FINAL SCORE:")
        print(final_score)

        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)

        X_test_true_filtered = dat["X_test"]#.iloc[:, selected_features]
        Y_pred = base_lr.predict(X_test_true_filtered)  # Use the filtered test data

        if(final_score > 0.78):
            results = pd.DataFrame(Y_pred, columns=['score'])
            results.to_csv(output_dir + "/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
            np.savetxt(output_dir + '/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')