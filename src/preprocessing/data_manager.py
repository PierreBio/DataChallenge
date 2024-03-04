import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro
from imblearn.over_sampling import SMOTE
import pickle

def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    if isinstance(data, pd.DataFrame):
        print(type(data))
    with open(filepath, 'rb') as handle:
        dat = pd.read_pickle(handle)
        print(dat.keys())
    return dat

def prepare_data(dat, test_size=0.5, random_state=42, use_pca=False, pca_components=0.95, winsorize_limits=(0.01, 0.01)):
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        dat['X_train'], dat['Y'], dat['S_train'],
        test_size=test_size, random_state=random_state, stratify=np.column_stack([dat['Y'], dat['S_train']])
    )

    for column in X_train.columns:
        X_train[column] = winsorize(X_train[column], limits=winsorize_limits)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights_dict = {class_label: weight for class_label, weight in zip(np.unique(Y_train), class_weights)}
    sample_weights = compute_sample_weight(class_weight=class_weights_dict, y=Y_train_resampled)

    X_test_pca = dat['X_test']
    if use_pca:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_resampled = pca.fit_transform(X_train_resampled)
        X_test = pca.transform(X_test)
        X_test_pca = pca.transform(dat['X_test'])

    return X_train_resampled, X_test, X_test_pca, Y_train_resampled, Y_test, S_train, S_test, class_weights, sample_weights

def check_normality_by_class(X, Y, feature_index=0, max_classes=28):
    df = pd.DataFrame({
        'Value': X[:, feature_index],
        'Class': Y
    })

    classes = np.unique(Y)[:max_classes]

    (min_quantile, max_quantile), (min_value, max_value) = (-4, 4),(-2, 2)
    for cls in np.unique(Y):
        stats.probplot(X[Y == cls, feature_index], dist="norm", plot=plt)
        plt.title(f'QQ-Plot for Class {cls}')
        plt.xlim(min_quantile, max_quantile)
        plt.ylim(min_value, max_value)
        plt.show()

    results = []
    for i, cls in enumerate(classes, 1):
        data_cls = df[df['Class'] == cls]['Value']
        median = np.median(data_cls)
        variance = np.var(data_cls)
        mean = np.mean(data_cls)
        min_val = np.min(data_cls)
        max_val = np.max(data_cls)

        results.append({
            'Class': cls,
            'Median': median,
            'Mean': mean,
            'Variance': variance,
            'Min': min_val,
            'Max': max_val
        })

    for result in results:
        print(f"Class {result['Class']}: Median = {result['Median']}, Mean = {result['Mean']}, Variance = {result['Variance']}, Min = {result['Min']}, Max = {result['Max']}")

        plt.tight_layout()
        plt.show()