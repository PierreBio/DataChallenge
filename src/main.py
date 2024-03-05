from src.preprocessing.data_manager import *
from src.models.model_factory import ModelFactory
from src.postprocessing.evaluator import *
from src.postprocessing.performance import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

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
    config = load_config('./config/config.json')

    dat = load_data('./data/data-challenge-student.pickle')
    X_train, X_test, X_test_pca, Y_train, Y_test, S_train, S_test, class_weights, sample_weights = prepare_data(dat)
    #X_train, X_test, Y_train, Y_test, S_train, S_test = prepare_data_resample(dat)
    # Initialisation du scaler
    #scaler = MinMaxScaler()
    # Ajustement du scaler sur les données d'entraînement uniquement
    #scaler.fit(X_train)

    # Transformation des ensembles d'entraînement et de test
    #X_train_scaled = scaler.transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    # encoder, decoder = train_denoising_autoencoder(X_train, 768, 128, config['neural_network'])
    # X_train_encoded = encoder.predict(X_train)
    # X_test_encoded = encoder.predict(X_test)

    #model_type = 'logistic_regression'
    #model_config = config[model_type]
    #model = ModelFactory.get_model(model_type, X_train, Y_train, S_train, model_config, class_weights, sample_weights)
    #Y_pred = model.predict(X_test)  # Ensure to use encoded test data here

    #---------------------------------------------------------
    kmeans_augmenter = KMeansFeatures(n_clusters=28)  # You can adjust the number of clusters
    class_weights_dict = {class_label: weight for class_label, weight in zip(np.unique(Y_train), class_weights)}
    base_lr = LogisticRegression(solver='lbfgs', max_iter=5000, C=0.1, class_weight= class_weights_dict)
    ovr_classifier = OneVsRestClassifier(base_lr)

    # Build the pipeline
    pipeline = Pipeline([
        ('kmeans', kmeans_augmenter),
        ('classifier', ovr_classifier)
    ])

    # Now, you can fit your pipeline to the training data
    pipeline.fit(X_train, Y_train)

    # And predict on your test set
    Y_pred = pipeline.predict(X_test)
    #---------------------------------------------------------

    #base_lr = LogisticRegression(solver='lbfgs', max_iter=5000)
    # Create the OneVsRestClassifier
    #model = OneVsRestClassifier(base_lr)

    # Train the model
    #model .fit(X_train, Y_train)

    # Predict on the test set
    #Y_pred = model .predict(X_test)
    #---------------------------------------------------------

    # Evaluation
    eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
    print(final_score)

    X_test_true = dat["X_test"]
    Y_pred = pipeline.predict(X_test_true)  # Ensure to use encoded test data here

    results = pd.DataFrame(Y_pred, columns=['score'])
    results.to_csv("./results/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
    np.savetxt('./results/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')
    update_performance_record('pipeline', final_score, pipeline)