from src.preprocessing.data_manager import load_data, prepare_data
from src.models.model_factory import ModelFactory
from src.postprocessing.evaluator import *

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    config = load_config('./config/config.json')

    dat = load_data('./data/data-challenge-student.pickle')
    X_train, X_test, Y_train, Y_test, S_train, S_test = prepare_data(dat)

    model_type = 'neural_network'
    model_params = config[model_type]
    model = ModelFactory.get_model(model_type, X_train, Y_train, **model_params)

    Y_pred = model.predict(X_test)

    eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
    print(final_score)

    X_test_true = dat['X_test']
    S_test_true = dat['S_test']
    y_test_pred = model.predict(X_test_true)
    results = pd.DataFrame(y_test_pred, columns=['score'])
    results.to_csv("./results/Data_Challenge_MDI_341.csv", header=None, index=None)
