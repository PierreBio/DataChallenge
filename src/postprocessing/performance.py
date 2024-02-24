import pandas as pd
import json
from datetime import datetime

def update_performance_record(model_type, final_score, model):
    filename = './results/Data_Challenge_Performances.csv'

    with open('./config/config.json', 'r') as file:
        config = json.load(file)

    model_params = config.get(model_type, {})
    record = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': model_type,
        'final_score': final_score,
    }

    if model is not None and hasattr(model, 'best_epoch'):
        record['best_epoch'] = model.best_epoch

    record.update(model_params)

    try:
        results_df = pd.read_csv(filename, sep=';')
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=record.keys())

    new_df = pd.DataFrame([record])
    results_df = pd.concat([results_df, new_df], ignore_index=True)
    results_df = results_df.sort_values(by='final_score', ascending=False)
    results_df = results_df.reindex(columns=(results_df.columns.union(new_df.columns, sort=None)))

    results_df.to_csv(filename, sep=';', index=False)