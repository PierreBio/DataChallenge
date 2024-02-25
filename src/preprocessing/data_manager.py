import pandas as pd
from sklearn.model_selection import train_test_split

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

def prepare_data(dat, test_size=0.5, random_state=42):
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(dat['X_train'], dat['Y'], dat['S_train'], test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test, S_train, S_test