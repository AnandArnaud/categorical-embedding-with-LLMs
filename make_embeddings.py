from dirty_cat import datasets

import gc
import torch
import time
import csv

from functions.function import initialize_bert
from functions.function import get_embeddings
from functions.utils import embedding_to_pt
from functions.data_loader import load_datasets

all_datasets = load_datasets()

model, tokenizer = initialize_bert()

for dataset_name, dataset in all_datasets.items():
    X = dataset.X
    start = time.time()
    embedding_1 = get_embeddings(X, model, tokenizer, "val")
    file_name = dataset_name + "_val" + "_embeddings.pt"
    embedding_to_pt(embedding_1, file_name)
    end = time.time()
    with open('embeddings_comp_time.csv', mode='a') as temps_fichier:
        temps_writer = csv.writer(temps_fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        temps_writer.writerow([dataset_name + "_val", end - start])

    del(embedding_1)
    gc.collect()

    start = time.time()
    embedding_2 = get_embeddings(X, model, tokenizer,"val_col")
    embedding_to_pt(embedding_2, dataset_name + "_val_col" + "_embeddings.pt")
    end = time.time()
    with open('embeddings_comp_time.csv', mode='a') as temps_fichier:
        temps_writer = csv.writer(temps_fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        temps_writer.writerow([dataset_name + "_val_col", end - start])

    del(embedding_2)
    gc.collect()

    start = time.time()
    embedding_3 = get_embeddings(X, model, tokenizer, "val_col_sep")
    embedding_to_pt(embedding_3, dataset_name + "_val_col_sep" + "_embeddings.pt")
    end = time.time()
    with open('embeddings_comp_time.csv', mode='a') as temps_fichier:
        temps_writer = csv.writer(temps_fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        temps_writer.writerow([dataset_name + "_val_col_sep", end - start])

    del(embedding_3)
    gc.collect()