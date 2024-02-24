import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def load_data(filepath):
    with open(filepath, 'rb') as handle:
        dat = pd.read_pickle(handle)
    return dat

def prepare_data(dat, test_size=0.3, random_state=42):
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(dat['X_train'], dat['Y'], dat['S_train'], test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test, S_train, S_test
