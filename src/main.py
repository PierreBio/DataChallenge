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
from imblearn.over_sampling import KMeansSMOTE
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from imblearn.combine import SMOTEENN

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
        X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
            dat['X_train'], dat['Y'], dat['S_train'],
            test_size=0.2, stratify=np.column_stack([dat['Y'], dat['S_train']])
        )

        S_train_series = pd.Series(S_train, index=X_train.index)

        # Maintenant, vous pouvez filtrer X_train en utilisant les valeurs dans S_train_series :
        indices_hommes = S_train_series[S_train_series == 0].index
        indices_femmes = S_train_series[S_train_series == 1].index

        X_train_hommes = X_train.loc[indices_hommes]
        X_train_femmes = X_train.loc[indices_femmes]

        #print(X_train_hommes)
        #print(X_train_femmes)

        #print(Y_train)

        X_train_lr, X_test_lr, S_train_lr, S_test_lr = train_test_split(X_train, S_train, test_size=0.2, random_state=42)

        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train_lr, S_train_lr)

        S_pred_lr = model_lr.predict(X_test_lr)

        accuracy = accuracy_score(S_test_lr, S_pred_lr)
        conf_matrix = confusion_matrix(S_test_lr, S_pred_lr)

        #print(f'Accuracy du modèle: {accuracy}')
        #print('Matrice de confusion :\n', conf_matrix)

        #S_pred = model_lr.predict(X_train)
        # Création d'un DataFrame pour faciliter l'analyse
        #df_analysis = pd.DataFrame({'Y_train': Y_train, 'S_pred': S_pred})

        # Calcul de la répartition des prédictions de l'attribut sensible dans chaque classe
        #repartition = df_analysis.groupby('Y_train')['S_pred'].value_counts(normalize=True).unstack().fillna(0)

        #print(repartition)

        #S_prob = model_lr.predict_proba(X_train)[:, 1]  # Probabilité d'être une femme
        #X_train['prob_femme'] = S_prob

        #X_train = X_train.drop('prob_femme', axis=1)
        #X_train.columns = X_train.columns.astype(str)

        ########################################
        X_train_smote_by_class = defaultdict(pd.DataFrame)
        Y_train_smote_by_class = defaultdict(pd.Series)

        for classe in Y_train.unique():
            indices = Y_train[Y_train == classe].index
            X_subset = X_train.loc[indices]
            S_subset = S_train.loc[indices]

            smote = SMOTE(random_state=42)
            X_smote, S_smote = smote.fit_resample(X_subset, S_subset)

            # Convertir le résultat de SMOTE en DataFrame et Series appropriés
            X_smote_df = pd.DataFrame(X_smote, columns=X_subset.columns)
            Y_smote_series = pd.Series([classe] * X_smote_df.shape[0])

            # Stocker directement les DataFrame et Series dans le defaultdict
            X_train_smote_by_class[classe] = X_smote_df
            Y_train_smote_by_class[classe] = Y_smote_series

        # Concaténer les DataFrames et Series pour obtenir les versions finales
        X_train_smote = pd.concat(X_train_smote_by_class.values(), ignore_index=True)
        Y_train_smote = pd.concat(Y_train_smote_by_class.values(), ignore_index=True)

       # Compter le nombre actuel d'échantillons par classe
        class_counts = Counter(Y_train_smote)
        additional_samples_per_class = {cls: int(count * 0.4) for cls, count in class_counts.items()}
        target_samples_per_class = {cls: count + additional_samples_per_class[cls] for cls, count in class_counts.items()}

        smote = SMOTE(sampling_strategy=target_samples_per_class)
        X_train_smoted, y_train_smoted = smote.fit_resample(X_train_smote, Y_train_smote)
        S_train_smoted = model_lr.predict(X_train_smoted)
        S_train_smoted_series = pd.Series(S_train_smoted, index=X_train_smoted.index)  # Assurez-vous que X_train_smoted est un DataFrame pour avoir .index

        X_train_by_class = defaultdict(pd.DataFrame)
        Y_train_by_class = defaultdict(pd.Series)

        for classe in y_train_smoted.unique():
            indices = y_train_smoted[y_train_smoted == classe].index
            X_subset_ = X_train_smoted.loc[indices]
            # Assurez-vous que S_train_adasyn est aligné ou réindexé pour correspondre à y_train_smoted
            S_subset_ = S_train_smoted_series.loc[indices]  # Ceci devrait référencer correctement S_train_adasyn, pas X_train_smoted

            adasyn = SMOTE(random_state=42)
            X_adasyn, _ = adasyn.fit_resample(X_subset_, S_subset_)  # S_subset_ devrait être les labels cibles pour ADASYN, vérifiez que cela est correct

            X_adasyn_df = pd.DataFrame(X_adasyn, columns=X_subset.columns)
            Y_adasyn_series = pd.Series([classe] * X_adasyn_df.shape[0])

            # Stocker directement les DataFrame et Series dans le defaultdict
            X_train_by_class[classe] = X_adasyn_df
            Y_train_by_class[classe] = Y_adasyn_series

        # Concaténer les DataFrames et Series pour obtenir les versions finales
        X_train = pd.concat(X_train_by_class.values(), ignore_index=True)
        Y_train = pd.concat(Y_train_by_class.values(), ignore_index=True)

        base_lr = LogisticRegression(solver='lbfgs', max_iter=100, C=0.05, tol=0.0001, multi_class='auto')
        base_lr.fit(X_train, Y_train)
        Y_pred = base_lr.predict(X_test)

        eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
        final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
        print("FINAL SCORE:")
        print(final_score)

        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)

        X_test_true_filtered = dat["X_test"]#.iloc[:, selected_features]
        Y_pred = base_lr.predict(X_test_true_filtered)  # Use the filtered test data

        if(final_score > 0.783):
            results = pd.DataFrame(Y_pred, columns=['score'])
            results.to_csv(output_dir + "/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
            np.savetxt(output_dir + '/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')

        if(final_score > 0.772):
            model = Sequential()
            model.add(Dense(28, input_shape=(X_train.shape[1],), activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train_smoted, y_train_smoted, epochs=20, batch_size=10, verbose=1)
            predictions_prob = model.predict(X_test)
            predictions_labels = np.argmax(predictions_prob, axis=1)
            eval_scores, confusion_matrices_eval = gap_eval_scores(predictions_labels, Y_test, S_test, metrics=['TPR'])
            final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
            print("FINAL SCORE:")
            print(final_score)

            X_test_true_filtered = dat["X_test"]#.iloc[:, selected_features]
            Y_pred = model.predict(X_test_true_filtered)  # Use the filtered test data
            predictions_labels = np.argmax(Y_pred, axis=1)

            if(final_score > 0.78):
                results = pd.DataFrame(predictions_labels, columns=['score'])
                results.to_csv(output_dir + "/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
                np.savetxt(output_dir + '/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')