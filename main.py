import pandas as pd
import gc

from functions.utils import save_scores_to_csv
from functions.data_loader import load_datasets
from functions.SuperVectorizer_baseline import run_baseline_model
from functions.function import run_model, extract_embeddings


all_datasets = load_datasets()

for dataset_name, dataset in all_datasets.items():
    
    baseline_results_df = run_baseline_model(dataset, dataset_name)
    save_scores_to_csv(baseline_results_df, "result.csv")

    delete(baseline_results_df)
    gc.collect()

    embeddings = extract_embeddings("val", "last", dataset_name)
    result = run_model(dataset, dataset_name, embeddings, "val")
    save_scores_to_csv(result, "result.csv")

    delete(result)
    gc.collect()

    embeddings = extract_embeddings("val", "last_four", dataset_name)
    result = run_model(dataset, dataset_name, embeddings, "val")
    save_scores_to_csv(result, "result.csv")

    delete(result)
    gc.collect()

    embeddings = extract_embeddings("val_col", "last", dataset_name)
    result = run_model(dataset, dataset_name, embeddings, "val_col")
    save_scores_to_csv(result, "result.csv")

    delete(result)
    gc.collect()

    embeddings = extract_embeddings("val_col", "last_four", dataset_name)
    result = run_model(dataset, dataset_name, embeddings, "val_col")
    save_scores_to_csv(result, "result.csv")

    delete(result)
    gc.collect()

    embeddings = extract_embeddings("val_col_sep", "last", dataset_name)
    result = run_model(dataset, dataset_name, embeddings, "val_col_sep")
    save_scores_to_csv(result, "result.csv")

    delete(result)
    gc.collect()

    embeddings = extract_embeddings("val_col_sep", "last_four", dataset_name)
    result = run_model(dataset, dataset_name, embeddings, "val_col_sep")
    save_scores_to_csv(result, "result.csv")

    delete(result)
    gc.collect()