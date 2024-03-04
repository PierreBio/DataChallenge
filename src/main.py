from src.preprocessing.data_manager import *
from src.models.model_factory import ModelFactory
from src.postprocessing.evaluator import *
from src.postprocessing.performance import *

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    config = load_config('./config/config.json')

    dat = load_data('./data/data-challenge-student.pickle')
    X_train, X_test, X_test_pca, Y_train, Y_test, S_train, S_test, class_weights, sample_weights = prepare_data(dat)
    #check_normality_by_class(X_train.to_numpy(), Y_train, feature_index=0)

    model_type = 'logistic_regression'
    model_config = config[model_type]
    model = ModelFactory.get_model(model_type, X_train, Y_train, S_train, model_config, class_weights, sample_weights)

    Y_pred = model.predict(X_test)

    # Evaluation
    eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1 - eval_scores['TPR_GAP'])) / 2
    print(final_score)

    results = pd.DataFrame(Y_pred, columns=['score'])
    results.to_csv("./results/Data_Challenge_MDI_341_" + str(final_score) + ".csv", header=None, index=None)
    np.savetxt('./results/y_test_challenge_student' + str(final_score) + '.txt', Y_pred, delimiter=',')
    update_performance_record(model_type, final_score, model)