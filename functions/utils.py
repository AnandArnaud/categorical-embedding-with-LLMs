import pandas as pd
import os
import torch

from dirty_cat import datasets, TableVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

def get_pipeline(y):
    target_type = y.dtype.name
    if target_type == 'object':
        pipeline = make_pipeline(
            TableVectorizer(auto_cast=True),
            HistGradientBoostingClassifier()
        )
    else:
        pipeline = make_pipeline(
            TableVectorizer(auto_cast=True),
            HistGradientBoostingRegressor()
        )
    return pipeline


def get_scoring(y):
    target_type = y.dtype.name
    if 'float' in target_type or 'int' in target_type:
        return ['r2', "neg_mean_squared_error", 'neg_mean_absolute_error']
    elif 'object' in target_type:
        if y.nunique()>2:
            return ['accuracy', 'roc_auc_ovr', 'f1_weighted']
        else:
            return ['accuracy', 'roc_auc', 'f1']
    else:
        raise ValueError("Unknown target type")


def embedding_to_pt(embedding, file_name):
    results_dir = 'embeddings'
    file_path = os.path.join(results_dir, file_name)
    torch.save(embedding, file_path)


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
        df.to_csv(file_path, mode='a', header=False, index=False)


def transform_dict(d):
    new_dict = {}
    for key, value in d.items():
        if key.startswith('test') or key.startswith('fit') or key.startswith("score"):
            for i in range(1, len(value) + 1):
                new_key = f"{key}_{i}"
                new_dict[new_key] = [value[i-1]]
    return new_dict