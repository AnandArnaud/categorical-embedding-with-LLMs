import pandas as pd
import os

from dirty_cat import datasets, SuperVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

def get_pipeline(target_type):
    if target_type == 'object':
        pipeline = make_pipeline(
            SuperVectorizer(auto_cast=True),
            HistGradientBoostingClassifier()
        )
    else:
        pipeline = make_pipeline(
            SuperVectorizer(auto_cast=True),
            HistGradientBoostingRegressor()
        )
    return pipeline


def get_scoring(target_type):
    if 'float' in target_type or 'int' in target_type:
        return 'r2'
    elif 'object' in target_type:
        return 'accuracy'
    else:
        raise ValueError("Unknown target type")


def save_scores_to_csv(df, file_name):
    """
    Saves the given dataframe to a CSV file with the given file name.
    If the file already exists, it will be overwritten.
    """
    
    results_dir = 'results'
    file_path = os.path.join(results_dir, file_name)

    if not os.path.isfile(file_path):
        df.to_csv(file_path, header=True, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)