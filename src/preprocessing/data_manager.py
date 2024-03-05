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

def prepare_data(dat, test_size=0.5, random_state=42, use_pca=False, pca_components=0.95, winsorize_limits=(0.01, 0.01), interpolate = True):
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        dat['X_train'], dat['Y'], dat['S_train'],
        test_size=test_size, random_state=random_state, stratify=np.column_stack([dat['Y'], dat['S_train']])
    )
    #check_normality_by_class(X_train.to_numpy(), Y_train, feature_index=0)
    #result = calculate_qq_divergence(X_train, Y_train)
    #print(result)
    explore_data(X_train, Y_train, S_train)
    X_train = pd.DataFrame(X_train)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_train)
    X_norm = sklearn.preprocessing.normalize(X_normalized, norm='l2')

    # Choose the number of clusters
    n_clusters = 28
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_norm)

    # Get the cluster assignments for each embedding
    cluster_labels = kmeans.labels_

    # Reduce the dimensionality for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(X_normalized)

    # Plot the reduced data with cluster assignments
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', marker='.')
    plt.colorbar()
    plt.show()

    #X_train = apply_winsorize_limits(X_train, Y_train, winsorize_limits_by_class)
    #if interpolate:
    #    X_train = interpolate_extremes(X_train)
    #else:

    X_train = pd.DataFrame(X_train)
    for column in X_train.columns:
        X_train[column] = winsorize(X_train[column], limits=winsorize_limits)
    #check_normality_by_class(X_train.to_numpy(), Y_train, feature_index=27)

    #X_train = apply_boxcox(X_train)

    #X_train, bins = apply_quantile_binning(X_train, 10)
    # Apply the same binning to the test set
    #X_test_binned = pd.DataFrame()
    #for i, column in enumerate(X_test.columns):
    #    X_test_binned[column] = pd.cut(X_test[column], bins=bins[i], labels=False, include_lowest=True)
    #X_test = X_test_binned

    #scaler = RobustScaler()
    #sX_train = scaler.fit_transform(X_train)
    # Make sure to transform the test set with the same scaler
    #sX_test = scaler.transform(X_test)

    #X_train, yeo_johnson_transformer = apply_yeo_johnson_transformation(X_train)
    # Transform the test set using the fitted transformer
    #X_test = yeo_johnson_transformer.transform(X_test)
    #X_test = pd.DataFrame(X_test, columns=X_train.columns)  # Ensure column names are consistent

    #X_train, Y_train = remove_outliers_isolation_forest(X_train, Y_train, contamination=0.001, random_state=random_state)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights_dict = {class_label: weight for class_label, weight in zip(np.unique(Y_train), class_weights)}
    sample_weights = compute_sample_weight(class_weight=class_weights_dict, y=Y_train_resampled)
    sample_weights = optimized_preprocessing(X_train, Y_train, S_train)

    adasyn = ADASYN(random_state=42)
    from collections import Counter
    print('Distribution des classes avant ADASYN:', Counter(Y_train_resampled))
    # Application de ADASYN sur vos données d'entraînement
    X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train_resampled, Y_train_resampled)
    # Vérification de la distribution des classes après suréchantillonnage
    print('Distribution des classes après ADASYN:', Counter(Y_train_resampled))

    X_test_pca = dat['X_test']
    if use_pca:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_resampled = pca.fit_transform(X_train_resampled)
        X_test = pca.transform(X_test)
        X_test_pca = pca.transform(dat['X_test'])

    return X_train_resampled, X_test, X_test_pca, Y_train_resampled, Y_test, S_train, S_test, class_weights, sample_weights

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

import numpy as np
from sklearn.utils import resample

def prepare_data_resample(dat, test_size=0.5, random_state=42):
    # Division initiale des données
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        dat['X_train'], dat['Y'], dat['S_train'],
        test_size=test_size, random_state=random_state, stratify=np.column_stack([dat['Y'], dat['S_train']])
    )

    train_data = pd.DataFrame(X_train)
    train_data['Y'] = Y_train
    train_data['S'] = S_train

    # Inspecter les groupes et décider des groupes à suréchantillonner
    inspect_groups(X_train)
    inspect_groups(Y_train)
    group_counts = inspect_groups(S_train)

    # Performer le ré-échantillonnage basé sur les décisions
    resampled_data = perform_resampling(train_data, group_counts, min_threshold=50)

    # Mise à jour de X_train, Y_train, S_train avec les données ré-échantillonnées
    X_train = resampled_data.drop(['Y', 'S'], axis=1).values
    Y_train = resampled_data['Y'].values
    S_train = resampled_data['S'].values

    return X_train, X_test, Y_train, Y_test, S_train, S_test

def perform_resampling(train_data, group_counts, min_threshold=50):
    """
    Effectue le suréchantillonnage des groupes sous-représentés basé sur un seuil minimum.
    """
    resampled_data = train_data.copy()
    for group, count in group_counts.items():
        if count < min_threshold:
            # Identifier les données du groupe sous-représenté
            group_data = train_data[train_data['S'] == group]

            # Suréchantillonner ce groupe jusqu'à atteindre le seuil minimum
            group_upsampled = resample(group_data,
                                       replace=True,
                                       n_samples=min_threshold,
                                       random_state=42)

            # Combiner avec le reste des données
            resampled_data = pd.concat([resampled_data, group_upsampled])

    return resampled_data

def perform_balancing(train_data):
    """
    Équilibre les groupes en suréchantillonnant le groupe sous-représenté.
    """
    # Séparer les données par groupe
    group_0_data = train_data[train_data['S'] == 0]
    group_1_data = train_data[train_data['S'] == 1]

    # Suréchantillonnage de group_1_data
    group_1_upsampled = resample(group_0_data,
                                 replace=False,  # échantillonnage avec remplacement
                                 n_samples=group_1_data.shape[0] ,  # nombre cible d'échantillons
                                 random_state=42)  # pour la reproductibilité

    # Si nécessaire, sous-échantillonnage de group_0_data pour équilibrer (optionnel)
    # group_0_downsampled = resample(group_0_data,
    #                                replace=False,
    #                                n_samples=target_occurrences,
    #                                random_state=42)

    # Dans cet exemple, nous nous concentrons sur le suréchantillonnage de group_1

    balanced_data = pd.concat([group_0_data, group_1_upsampled])
    print(balanced_data)
    return balanced_data

def inspect_groups(S_train):
    """
    Inspecte et affiche la distribution des groupes dans S_train.
    """
    from collections import Counter

    # Compter les occurrences de chaque groupe
    group_counts = Counter(S_train)

    # Afficher la distribution des groupes
    for group, count in group_counts.items():
        print(f"Groupe {group}: {count} occurrences")

    return group_counts

def check_normality_by_class(X, Y, feature_index=0, max_classes=28):
    df = pd.DataFrame({
        'Value': X[:, feature_index],
        'Class': Y
    })

    classes = np.unique(Y)[:max_classes]

    (min_quantile, max_quantile), (min_value, max_value) = (-3.5, 3.5),(-1.5, 0.7)
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

def apply_winsorizing_specific_to_class_with_values(X, y, winsorize_limits_by_class):
    X_winsorized = X.copy()

    for class_id, limits in winsorize_limits_by_class.items():
        # Sélectionner les indices pour la classe actuelle
        indices = y == class_id

        # Appliquer le Winsorizing à chaque caractéristique pour les indices sélectionnés
        for column in X.columns:
            # Ici, nous utilisons directement les limites comme des valeurs de coupure, pas comme des percentiles
            data = X.loc[indices, column]
            # Les limites sont définies en termes de valeurs réelles, et non de percentiles
            data_winsorized = winsorize(data, limits=(limits[0], limits[1]))
            X_winsorized.loc[indices, column] = data_winsorized

    return X_winsorized

def calculate_percentiles_for_limits(X, Y, winsorize_limits_by_class, feature_index):
    percentile_limits_by_class = {}

    for cls, (lower_limit, upper_limit) in winsorize_limits_by_class.items():
        # Sélectionner les données pour la classe et la caractéristique actuelles
        class_data = X[Y == cls, feature_index]

        # Trouver les indices des valeurs les plus proches des limites spécifiées
        sorted_data = np.sort(class_data)
        lower_idx = np.searchsorted(sorted_data, lower_limit, side='left')
        upper_idx = np.searchsorted(sorted_data, upper_limit, side='right') - 1  # Ajustement pour inclure la limite supérieure

        # Calculer les percentiles
        lower_percentile = 100.0 * lower_idx / len(class_data)
        upper_percentile = 100.0 * upper_idx / len(class_data)

        percentile_limits_by_class[cls] = (lower_percentile, upper_percentile)

    return percentile_limits_by_class

def apply_winsorize_limits(X, Y, winsorize_limits_by_class):
    # S'assurer que X est un numpy array pour faciliter l'indexation booléenne
    X_adjusted = X.copy()

    for cls, limits in winsorize_limits_by_class.items():
        # Identifier les échantillons appartenant à la classe actuelle
        indices = Y == cls

        # Parcourir toutes les caractéristiques par leurs noms
        for column in X.columns:
            # Appliquer le seuil bas
            lower_limit = limits[0]
            X_adjusted.loc[indices, column] = X.loc[indices, column].clip(lower=lower_limit)

            # Appliquer le seuil haut (clip applique à la fois le seuil bas et haut)
            upper_limit = limits[1]
            X_adjusted.loc[indices, column] = X.loc[indices, column].clip(upper=upper_limit)

    return X_adjusted

def calculate_percentiles_from_limits(X, Y, winsorize_limits_by_class):
    """
    Pour chaque classe, calcule les percentiles correspondant aux limites spécifiques données.

    :param X: Le tableau numpy des caractéristiques, avec des échantillons en lignes et des caractéristiques en colonnes.
    :param Y: Le tableau numpy des étiquettes de classe.
    :param winsorize_limits_by_class: Un dictionnaire des limites (valeurs numériques) pour chaque classe.
    :return: Un dictionnaire des percentiles (bas et haut) pour chaque classe.
    """
    percentile_limits_by_class = {}
    for cls, limits in winsorize_limits_by_class.items():
        # Sélectionner les données pour la classe actuelle
        class_data = X[Y == cls]

        # Calculer le percentile pour la limite inférieure et supérieure
        lower_percentile = np.percentile(class_data, limits[0], axis=0)
        upper_percentile = np.percentile(class_data, limits[1], axis=0)

        percentile_limits_by_class[cls] = (lower_percentile, upper_percentile)

    return percentile_limits_by_class

def calculate_qq_divergence(X, Y):
    """
    Calcule la divergence quantile-quantile pour chaque classe.

    :param X: Les caractéristiques des données, supposées être unidimensionnelles pour cette analyse.
    :param Y: Les étiquettes de classe.
    :return: Un dictionnaire avec les classes comme clés et les percentiles de divergence comme valeurs.
    """
    class_divergence = {}
    num_features = X.shape[1] if isinstance(X, np.ndarray) else X.shape[1]

    for cls in np.unique(Y):
        class_divergence[cls] = {}
        class_indices = Y == cls

        for feature_index in range(num_features):
            # Sélectionner les données pour la caractéristique actuelle de la classe actuelle
            class_data_feature = X[class_indices, feature_index] if isinstance(X, np.ndarray) else X.loc[class_indices, X.columns[feature_index]]

            # Calculer les quantiles théoriques et observés
            (osm, osr), _ = stats.probplot(class_data_feature, dist="norm")

            # Ici, vous pourriez calculer les écarts ou identifier les percentiles de divergence comme nécessaire
            # Ceci est juste un exemple pour obtenir osm et osr pour chaque caractéristique et classe
            class_divergence[cls][feature_index] = (osm, osr)

    return class_divergence

def interpolate_extremes(X, lower_percentile=1, upper_percentile=99):
    """
    Interpolate extreme values in data based on the central trend for each feature.
    """
    X_interpolated = X.copy()
    for column in X.columns:
        data = X[column]
        lower_bound, upper_bound = np.percentile(data, [lower_percentile, upper_percentile])

        # Fit a linear model to the central data
        central_data = data[(data >= lower_bound) & (data <= upper_bound)]
        # Generate a QQ plot for the central data
        osm, osr = stats.probplot(central_data, dist="norm")[0]
        osm = osm.reshape(-1, 1)  # Reshape osm for sklearn's LinearRegression
        model = LinearRegression().fit(osm, osr)

        # Predict and replace extreme values using the fitted model
        lower_extremes = data < lower_bound
        upper_extremes = data > upper_bound

        # Generate QQ plot data for extremes to predict their interpolated values
        if lower_extremes.any():
            osm_lower, _ = stats.probplot(data[lower_extremes], dist="norm")[0]
            predicted_lower = model.predict(osm_lower.reshape(-1, 1))
            X_interpolated.loc[lower_extremes, column] = predicted_lower

        if upper_extremes.any():
            osm_upper, _ = stats.probplot(data[upper_extremes], dist="norm")[0]
            predicted_upper = model.predict(osm_upper.reshape(-1, 1))
            X_interpolated.loc[upper_extremes, column] = predicted_upper

    return X_interpolated

def apply_boxcox(X):
    """
    Apply Box-Cox transformation to each feature in the dataset to make
    data distribution closer to normal.
    """
    X_boxcox = pd.DataFrame()
    for column in X.columns:
        # Shift the data to be strictly positive
        shifted_data = X[column] - X[column].min() + 1
        # Apply the Box-Cox transformation
        transformed_data, _ = stats.boxcox(shifted_data)
        # Store the transformed data
        X_boxcox[column] = transformed_data
    return X_boxcox

def apply_quantile_binning(X, n_bins):
    """
    Apply quantile binning to each feature in the dataset.
    """
    X_binned = pd.DataFrame()
    for column in X.columns:
        # Perform quantile binning
        X_binned[column], bins = pd.qcut(X[column], q=n_bins, labels=False, retbins=True, duplicates='drop')
    return X_binned, bins

def apply_square_root_transformation(X):
    """
    Apply square root transformation to each feature in the dataset.
    """
    X_sqrt_transformed = X.apply(np.sqrt)
    return X_sqrt_transformed

def remove_outliers_isolation_forest(X, y, contamination=0.05, random_state=42):
    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model and predict the outliers
    outliers = iso_forest.fit_predict(X)

    # Select only the normal data (non-outliers)
    mask = outliers != -1
    X_clean = X[mask]
    y_clean = y[mask]

    return X_clean, y_clean

def apply_yeo_johnson_transformation(X):
    """
    Apply the Yeo-Johnson transformation to each feature in the dataset.
    """
    transformer = PowerTransformer(method='yeo-johnson')
    X_transformed = transformer.fit_transform(X)
    return pd.DataFrame(X_transformed, columns=X.columns), transformer

def optimized_preprocessing(X_train, Y_train, S_train, method='reweighing'):
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


def whiten(X):
    """
    Whitening transformation on dataset X.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Eigenvalue decomposition of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)

    # Compute the inverse square root of the eigenvalues matrix
    eigen_vals_inv_sqrt = np.diag(1. / np.sqrt(eigen_vals + 1e-10))  # Add a small value to avoid division by zero

    # Compute the whitening matrix
    whitening_matrix = np.dot(eigen_vecs, np.dot(eigen_vals_inv_sqrt, eigen_vecs.T))

    # Apply the whitening matrix to the centered data
    X_whitened = np.dot(X_centered, whitening_matrix)

    return X_whitened

def add_noise(X, noise_factor=0.5):
    noisy_data = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return np.clip(noisy_data, 0., 1.)

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
