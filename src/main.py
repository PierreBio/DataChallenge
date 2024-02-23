import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pickle

from src.evaluator import *

if __name__ == "__main__":
    with open('./data/data-challenge-student.pickle', 'rb') as handle:
        # dat = pickle.load(handle)
        dat = pd.read_pickle(handle)

    X = dat['X_train']
    Y = dat['Y']
    S = dat['S_train']

    # Train the logistic regression
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=0.3, random_state=42)
    clf = LogisticRegression(random_state=0, max_iter=5000).fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])

    final_score = (eval_scores['macro_fscore']+ (1-eval_scores['TPR_GAP']))/2
    print(final_score)

    # Load the "true" test data
    X_test = dat['X_test']
    S_test = dat['S_test']

    # Classify the provided test data with you classifier
    y_test = clf.predict(X_test)
    results=pd.DataFrame(y_test, columns= ['score'])

    results.to_csv("./results/Data_Challenge_MDI_341.csv", header = None, index = None)
    # np.savetxt('y_test_challenge_student.txt', y_test, delimiter=',')