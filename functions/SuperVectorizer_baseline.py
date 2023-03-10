import pandas as pd
import numpy as np
import time
import datetime

from sklearn.model_selection import cross_val_score

from functions.utils import get_pipeline, get_scoring


def run_baseline_model(all_datasets):
    results = []

    for dataset_name, dataset in all_datasets.items():
        start = time.time()
        X = dataset.X
        y = dataset.y

        target_type = y.dtype.name
        pipeline = get_pipeline(target_type)
        scoring = get_scoring(target_type)

        print(dataset_name)

        scores = cross_val_score(pipeline, X, y, scoring=scoring, n_jobs=-1)
        print(np.mean(scores))
        end = time.time()
        time_delta = datetime.timedelta(seconds = end-start)

        result = pd.DataFrame({
            'dataset_name': [dataset_name],
            "strategy" : ["SuperVectorizer"],
            'mean_score': [np.mean(scores)],
            'std_score': [np.std(scores)],
            "compute_time" : [time_delta]
        })
        results.append(result)

    results_df = pd.concat(results)

    return results_df
