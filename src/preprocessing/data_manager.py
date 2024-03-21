from collections import Counter, defaultdict
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    if isinstance(data, pd.DataFrame):
        print(type(data))
    with open(filepath, 'rb') as handle:
        dat = pd.read_pickle(handle)
        print(dat.keys())
    return dat

def prepare_data(dat, test_size=0.2, random_state=42):
    #explore_data(dat['X_train'], dat['Y'], dat['S_train'])

    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        dat['X_train'], dat['Y'], dat['S_train'],
        test_size=test_size, stratify=np.column_stack([dat['Y'], dat['S_train']])
    )

    X_train_smote_by_class = defaultdict(pd.DataFrame)
    Y_train_smote_by_class = defaultdict(pd.Series)

    for classe in Y_train.unique():
        indices = Y_train[Y_train == classe].index
        X_subset = X_train.loc[indices]
        S_subset = S_train.loc[indices]

        smote = SMOTE(random_state=42)
        X_smote, S_smote = smote.fit_resample(X_subset, S_subset)

        X_smote_df = pd.DataFrame(X_smote, columns=X_subset.columns)
        Y_smote_series = pd.Series([classe] * X_smote_df.shape[0])

        X_train_smote_by_class[classe] = X_smote_df
        Y_train_smote_by_class[classe] = Y_smote_series

    X_train_smote = pd.concat(X_train_smote_by_class.values(), ignore_index=True)
    Y_train_smote = pd.concat(Y_train_smote_by_class.values(), ignore_index=True)

    class_counts = Counter(Y_train_smote)
    additional_samples_per_class = {cls: int(count * 0.2) for cls, count in class_counts.items()}
    target_samples_per_class = {cls: count + additional_samples_per_class[cls] for cls, count in class_counts.items()}

    smote = SMOTE(sampling_strategy=target_samples_per_class)
    X_train_final, Y_train_final = smote.fit_resample(X_train_smote, Y_train_smote)

    return X_train_final, X_test, Y_train_final, Y_test, S_train, S_test

def explore_data(X_train, Y_train, S_train=None, max_features=20):
    sns.set(style="whitegrid")

    features = X_train.columns[:max_features]
    n_features = len(features)
    n_rows = min(n_features, 4)
    n_cols = int(np.ceil(n_features / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i, feature in enumerate(features):
        if n_features == 1:
            ax = axs
        else:
            ax = axs[i]
        sns.histplot(X_train[feature], kde=True, ax=ax)
        ax.set_title(f'Distribution: {feature}')

    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 4))
    sns.histplot(Y_train, kde=True, bins=len(pd.unique(Y_train)))
    plt.title('Distribution of Target Variable (Y_train)')
    plt.show()

    if S_train is not None:
        plt.figure(figsize=(8, 4))
        sns.histplot(S_train, kde=True, bins=len(pd.unique(S_train)))
        plt.title('Distribution of Sensitive Attribute (S_train)')
        plt.show()