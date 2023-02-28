# import pandas as pd
# import numpy as np

# from dirty_cat import datasets
# from tqdm import tqdm


# def load_datasets_from_dirty_cat():
#     all_datasets = {}
#     for func_name in dir(datasets):
#         if func_name.startswith('fetch_'):
#             if func_name not in ['fetch_road_safety', 'fetch_traffic_violations']:
#                 dataset_name = func_name.replace('fetch_', '')
#                 dataset = getattr(datasets, func_name)()
#                 all_datasets[dataset_name] = dataset
#     print("datasets from dirty_cat loaded")
#     return all_datasets

import pandas as pd
from dirty_cat import datasets


def load_datasets_from_dirty_cat():
    all_datasets = {}
    for func_name in dir(datasets):
        if func_name.startswith('fetch_') and func_name not in ['fetch_road_safety', 'fetch_traffic_violations']:
            dataset_name = func_name.replace('fetch_', '')
            dataset = getattr(datasets, func_name)()
            all_datasets[dataset_name] = dataset
    print(f"{len(all_datasets)} datasets from dirty_cat loaded")
    return all_datasets
