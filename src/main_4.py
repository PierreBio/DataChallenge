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
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import PolynomialFeatures

class KMeansFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def fit(self, X, y=None):
        # Fit KMeans to the data
        self.kmeans.fit(X)
        return self  # Return the object itself to allow chaining

    def transform(self, X):
        # Predict cluster assignments for samples
        clusters = self.kmeans.predict(X)
        # Augment the features with cluster assignments
        return np.hstack((X, clusters.reshape(-1, 1)))

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
        X_train = dat['X_train']
        Y_train = dat['Y']
        S_train = dat['S_train']

        S_train_series = pd.Series(S_train, index=X_train.index)

        # Maintenant, vous pouvez filtrer X_train en utilisant les valeurs dans S_train_series :
        indices_hommes = S_train_series[S_train_series == 0].index
        indices_femmes = S_train_series[S_train_series == 1].index

        X_train_hommes = X_train.loc[indices_hommes]
        X_train_femmes = X_train.loc[indices_femmes]


        X_train_smote_by_class = defaultdict(pd.DataFrame)
        Y_train_smote_by_class = defaultdict(pd.Series)
        S_train_smote_by_class = defaultdict(pd.Series)

        for classe in Y_train.unique():
            indices = Y_train[Y_train == classe].index
            X_subset = X_train.loc[indices]
            S_subset = S_train.loc[indices]

            smote = SMOTE(random_state=42)
            X_smote, S_smote = smote.fit_resample(X_subset, S_subset)

            # Convertir le résultat de SMOTE en DataFrame et Series appropriés
            X_smote_df = pd.DataFrame(X_smote, columns=X_subset.columns)
            Y_smote_series = pd.Series([classe] * X_smote_df.shape[0])
            S_smote_series = pd.Series(S_smote.ravel())

            # Stocker directement les DataFrame et Series dans le defaultdict
            X_train_smote_by_class[classe] = X_smote_df
            Y_train_smote_by_class[classe] = Y_smote_series
            S_train_smote_by_class[classe] = S_smote_series

        # Concaténer les DataFrames et Series pour obtenir les versions finales
        X_train_smote = pd.concat(X_train_smote_by_class.values(), ignore_index=True)
        Y_train_smote = pd.concat(Y_train_smote_by_class.values(), ignore_index=True)
        S_train_smote = pd.concat(S_train_smote_by_class.values(), ignore_index=True)

        # Encoder Y_train et S_train en strings pour éviter toute confusion numérique
        Y_train_str = Y_train_smote.astype(str)
        S_train_str = S_train_smote.astype(str)

        # Créer la variable cible combinée en concaténant les valeurs de classe et de genre
        Y_S_combined = Y_train_str + "_" + S_train_str  # Par exemple, "classe1_genre0"

        smote = SMOTE(random_state=42)
        X_train_smote, Y_S_combined_smote = smote.fit_resample(X_train_smote, Y_S_combined)

        Y_train_smote = Y_S_combined_smote.str.split("_").str[0].astype(int)  # Ou le type original de Y_train
        S_train_smote = Y_S_combined_smote.str.split("_").str[1].astype(int)  # Ou le type original de S_train

        X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
            X_train_smote, Y_train_smote, S_train_smote,
            test_size=0.5, stratify=np.column_stack([Y_train_smote, S_train_smote])
        )

        '''
        config = load_config('./config/config.json')
        model_type = 'neural_network'
        model_config = config[model_type]
        model = ModelFactory.get_model(model_type, X_train, Y_train, S_train, model_config)

        Y_pred = model.predict(X_test)
        '''
        #poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        #X_poly = poly.fit_transform(X_train)

        base_lr = LogisticRegression(solver='lbfgs', max_iter=100, C=0.001, tol=0.00750, multi_class='multinomial')
        base_lr.fit(X_train, Y_train)
        Y_pred = base_lr.predict(X_test)

        eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
        final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
        print("TPR_GAP : ", (1 - eval_scores['TPR_GAP']))
        print("macro_fscore : ", eval_scores['macro_fscore'])
        print("FINAL SCORE:")
        print(final_score)

        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)

        X_test_true_filtered = dat["X_test"]#.iloc[:, selected_features]
        Y_pred = base_lr.predict(X_test_true_filtered)  # Use the filtered test data

        if(final_score > 0.785):
            results = pd.DataFrame(Y_pred, columns=['score'])
            results.to_csv(output_dir + "/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
            np.savetxt(output_dir + '/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')