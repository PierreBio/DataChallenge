import os
import shutil
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

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

    S_pred = model_lr.predict(X_train)
    # Création d'un DataFrame pour faciliter l'analyse
    df_analysis = pd.DataFrame({'Y_train': Y_train, 'S_pred': S_pred})

    # Calcul de la répartition des prédictions de l'attribut sensible dans chaque classe
    repartition = df_analysis.groupby('Y_train')['S_pred'].value_counts(normalize=True).unstack().fillna(0)

    #print(repartition)

    S_prob = model_lr.predict_proba(X_train)[:, 1]  # Probabilité d'être une femme
    X_train['prob_femme'] = S_prob

    X_train = X_train.drop('prob_femme', axis=1)
    X_train.columns = X_train.columns.astype(str)

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

    ########################################

    # Combinez tous les DataFrame stockés dans le defaultdict en un seul DataFrame
    #print(X_train_smote.shape)
    #print(Y_train_smote.shape)

    S_pred_smote = model_lr.predict(X_train_smote)
    # Création d'un DataFrame pour faciliter l'analyse
    df_analysis = pd.DataFrame({'Y_train': Y_train_smote, 'S_pred': S_pred_smote})
    # Calcul de la répartition des prédictions de l'attribut sensible dans chaque classe
    repartition = df_analysis.groupby('Y_train')['S_pred'].value_counts(normalize=True).unstack().fillna(0)
    print(repartition)

    base_lr = LogisticRegression(solver='lbfgs', max_iter=100, C=0.2, tol=100, multi_class='auto')
    base_lr.fit(X_train_smote, Y_train_smote)
    Y_pred = base_lr.predict(X_test)

    eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
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