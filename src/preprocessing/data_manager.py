import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    if isinstance(data, pd.DataFrame):
        print(type(data))
    with open(filepath, 'rb') as handle:
        dat = pd.read_pickle(handle)
        print(dat.keys())
    return dat

from imblearn.over_sampling import ADASYN
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn

def prepare_data(dat, test_size=0.5, random_state=42, winsorize_limits=(0.01, 0.01)):
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        dat['X_train'], dat['Y'], dat['S_train'],
        test_size=test_size, random_state=random_state, stratify=np.column_stack([dat['Y'], dat['S_train']])
    )

    #explore_data(X_train, Y_train, S_train)

    X_train = pd.DataFrame(X_train)

    for column in X_train.columns:
        lower_bound = X_train[column].quantile(0.01)
        upper_bound = X_train[column].quantile(0.99)
        X_train[column] = X_train[column].clip(lower=lower_bound, upper=upper_bound)

    for column in X_train.columns:
        X_train[column] = winsorize(X_train[column], limits=winsorize_limits)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights_dict = {class_label: weight for class_label, weight in zip(np.unique(Y_train), class_weights)}
    sample_weights = optimized_preprocessing(X_train, Y_train, S_train)

    X_train_resampled = X_train_resampled.astype('float32')
    Y_train_resampled = Y_train_resampled.astype('float32')

    return X_train_resampled, X_test, Y_train_resampled, Y_test, S_train, S_test, class_weights_dict, sample_weights

def explore_data(X_train, Y_train, S_train=None, max_features=20):
    sns.set(style="whitegrid")

    # Select a subset of features if there are too many
    features = X_train.columns[:max_features]
    n_features = len(features)
    n_rows = min(n_features, 4)  # Adjust based on your preference or screen resolution
    n_cols = int(np.ceil(n_features / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()  # Flatten to 1D array for easy iteration

    for i, feature in enumerate(features):
        if n_features == 1:  # If there's only one feature, axs is not an array
            ax = axs
        else:
            ax = axs[i]
        sns.histplot(X_train[feature], kde=True, ax=ax)
        ax.set_title(f'Distribution: {feature}')

    plt.tight_layout()
    plt.show()

    # Plot distribution of Y_train
    plt.figure(figsize=(8, 4))
    sns.histplot(Y_train, kde=True, bins=len(pd.unique(Y_train)))
    plt.title('Distribution of Target Variable (Y_train)')
    plt.show()

    # Plot distribution of S_train if it's provided
    if S_train is not None:
        plt.figure(figsize=(8, 4))
        sns.histplot(S_train, kde=True, bins=len(pd.unique(S_train)))
        plt.title('Distribution of Sensitive Attribute (S_train)')
        plt.show()

def optimized_preprocessing(X_train, Y_train, S_train):
    weights = calculate_bias_mitigation_weights(X_train, Y_train, S_train)
    return weights

def calculate_bias_mitigation_weights(X_train, Y_train, S_train):
    """
    Calculates weights to mitigate bias in training data.

    Parameters:
    - X_train: Features of the training data
    - Y_train: Target labels
    - S_train: Sensitive attributes

    Returns:
    - weights: A numpy array of weights for each instance in X_train
    """
    unique_classes = np.unique(Y_train)
    unique_sensitive_attrs = np.unique(S_train)
    weights = np.ones(len(Y_train))

    for class_ in unique_classes:
        for sensitive_attr in unique_sensitive_attrs:
            # Identify samples belonging to the current class and sensitive attribute group
            mask = (Y_train == class_) & (S_train == sensitive_attr)
            # Calculate the proportion of samples in the current group to all samples in the current class
            proportion = np.mean(mask) / np.mean(Y_train == class_)
            # Calculate the weight for the current group (inverse of proportion)
            weight_for_group = 1 / (proportion + 1e-8)  # Adding a small value to avoid division by zero

            # Assign the calculated weight to all samples in the current group
            weights[mask] = weight_for_group

    # Normalize the weights
    weights /= np.mean(weights)
    return weights

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def train_denoising_autoencoder(X_train_scaled, input_dim, encoding_dim, config):
    autoencoder, encoder, decoder = build_denoising_autoencoder(input_dim)

    # Configuration des callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=10, min_lr=0.0000001, verbose=1)

    # Entraînement de l'autoencoder
    autoencoder.fit(X_train_scaled, X_train_scaled,  # Les entrées et les cibles sont les mêmes pour un autoencoder
                    epochs=config['epochs'],
                    batch_size=4096,
                    validation_split=0.5,
                    callbacks=[early_stopping, reduce_lr])
    return encoder, decoder


def build_denoising_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    # Encodeur réduit
    x = Dense(256, activation='relu')(input_layer)  # Réduction de 512 à 256 unités
    x = Dropout(0.2)(x)  # Légère réduction du dropout
    encoded = Dense(64, activation='relu')(x)  # Réduction de 128 à 64 unités pour l'espace latent

    # Décodeur réduit
    x = Dense(256, activation='relu')(encoded)  # Correspond à la première couche de l'encodeur
    x = Dropout(0.2)(x)
    decoded = Dense(input_dim, activation='sigmoid')(x)  # Pas de couche supplémentaire de 512 unités

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    # Construction du décodeur (aucun changement ici car cela dépend de l'espace latent et de la sortie)
    decoder_input = Input(shape=(64,))
    _x = Dense(256, activation='relu')(decoder_input)
    _x = Dropout(0.2)(_x)
    decoder_output = Dense(input_dim, activation='sigmoid')(_x)
    decoder = Model(decoder_input, decoder_output)

    # Configuration de l'optimiseur avec un taux d'apprentissage initial
    initial_learning_rate = 0.1
    optimizer = Adam(learning_rate=initial_learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    return autoencoder, encoder, decoder
