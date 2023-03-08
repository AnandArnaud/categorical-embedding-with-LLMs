import pandas as pd
import requests
import sys

from dirty_cat import datasets


class CustomDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    

def import_colleges():
    url = "https://beachpartyserver.azurewebsites.net/VueBigData/DataFiles/Colleges.txt"
    response = requests.get(url)

    with open("../datasets/colleges.txt", "w") as f:
        f.write(response.text)


def fetch_colleges():
    # Charger les données
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/colleges.txt", delimiter="\t", index_col=0)
    df = df.dropna(subset=["Percent Pell Grant"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X = df.drop(columns=["Percent Pell Grant"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y = df.loc[:, "Percent Pell Grant"]
    # Créer l'objet CustomDataset
    colleges = CustomDataset(X, y)
    return colleges


def import_journal_influence():
    url = "https://raw.githubusercontent.com/FlourishOA/Data/master/estimated-article-influence-scores-2015.csv"
    response = requests.get(url)

    with open("../datasets/journal_influence.txt", "w") as f:
        f.write(response.text)


def fetch_journal_influence():
    # Charger les données
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/journal_influence.txt", delimiter=",", index_col=0)
    df = df.dropna(subset=["avg_cites_per_paper"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X = df.drop(columns=["avg_cites_per_paper"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y = df.loc[:, "avg_cites_per_paper"]
    # Créer l'objet CustomDataset
    journal_influence = CustomDataset(X, y)
    return journal_influence


def import_vancouver_employee():
    url = "https://opendata.vancouver.ca/explore/dataset/employee-remuneration-and-expenses-earning-over-75000/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true"
    response = requests.get(url)

    with open("../datasets/vancouver_employee.txt", "w") as f:
        f.write(response.text)


def fetch_vancouver_employee():
    # Charger les données
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/vancouver_employee.txt", delimiter=";")
    df = df.dropna(subset=["Remuneration"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X = df.drop(columns=["Remuneration"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y = df.loc[:, "Remuneration"]
    # Créer l'objet CustomDataset
    vancouver_employee = CustomDataset(X, y)
    return vancouver_employee


def load_datasets():
    all_datasets = {}
    for func_name in dir(datasets):
        if func_name.startswith('fetch_') and func_name not in ["fetch_open_payments", "fetch_medical_charge", "fetch_road_safety", "fetch_traffic_violations", "fetch_drug_directory", "fetch_figshare", "fetch_world_bank_indicator"]:
            dataset_name = func_name.replace('fetch_', '')
            print(dataset_name)
            dataset = getattr(datasets, func_name)()
            all_datasets[dataset_name] = dataset
    print(f"{len(all_datasets)} datasets from dirty_cat loaded")
    
    for func_name in dir(sys.modules[__name__]):
        if func_name.startswith("fetch_"):
            dataset_name = func_name.replace("fetch_", "")
            print(dataset_name)
            dataset = getattr(sys.modules[__name__], func_name)()
            all_datasets[dataset_name] = dataset
    print(f"Total dataset loaded: {len(all_datasets)}")
    return all_datasets
