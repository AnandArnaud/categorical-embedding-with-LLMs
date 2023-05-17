import pandas as pd
import numpy as np
import time
import datetime

from sklearn.model_selection import cross_validate

from functions.utils import get_pipeline, get_scoring, transform_dict


def run_baseline_model(dataset, dataset_name):

    start = time.time()
    X = dataset.X
    y = dataset.y

    pipeline = get_pipeline(y)
    scoring = get_scoring(y)

    print(dataset_name)

    cv_results = cross_validate(pipeline, X, y, scoring=scoring, return_train_score=True, n_jobs=-1)
    end = time.time()
    time_delta = datetime.timedelta(seconds = end-start)

    cv_results = transform_dict(cv_results)

    result = pd.DataFrame({
    'dataset_name': [dataset_name],
    "strategy" : ["TableVectorizer"],
    "compute_time" : [time_delta]})

    cv_results = pd.DataFrame.from_dict(cv_results)

    final_df = pd.concat([result, cv_results], axis=1)

    return final_df
