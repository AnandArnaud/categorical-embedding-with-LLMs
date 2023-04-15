import pandas as pd
import numpy as np
import time
import datetime

from sklearn.model_selection import cross_val_score

from functions.utils import get_pipeline, get_scoring


def run_baseline_model(dataset, dataset_name):

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
        "strategy" : ["TableVectorizer"],
        'mean_score': [np.mean(scores)],
        'std_score': [np.std(scores)],
        "compute_time" : [time_delta]
    })

    return result 
