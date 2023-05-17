import pandas as pd
import requests
import sys

from dirty_cat import datasets
from sklearn.model_selection import train_test_split


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


def fetch_adult():
    # Charger les données
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/adult.txt", delimiter=", ")
    df = df.dropna(subset=["Income"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X = df.drop(columns=["Income"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y = df.loc[:, "Income"]
    # Créer l'objet CustomDataset
    adult = CustomDataset(X, y)
    return adult


def fetch_house_sales():
    # Charger les données
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/house_sales.csv", delimiter=",")
    df = df.dropna(subset=["price"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X = df.drop(columns=["price"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y = df.loc[:, "price"]
    # Créer l'objet CustomDataset
    house_sales = CustomDataset(X, y)
    return house_sales


def fetch_kickstarter_projects():
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/kickstarter_projects.csv", encoding='latin1', delimiter=",")
    df = df.dropna(subset=["state "])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    valid_categories = ["successful", "canceled", "live", "undefined", "suspended", 'failed']
    df = df[df['state '].isin(valid_categories)]
    X1 = df.drop(columns=["state "])
    # Sélectionner uniquement la colonne target pour obtenir y
    y1 = df.loc[:, "state "]
    X, _, y, _ = train_test_split(X1, y1, train_size= 30000, stratify=y1, random_state=0)
    # Créer l'objet CustomDataset
    kickstarter_projects = CustomDataset(X, y)
    return kickstarter_projects


def fetch_met_objects():
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/MetObjects.csv", delimiter=",")
    df = df.drop(["Artist Wikidata URL", "Artist ULAN URL", "Link Resource", 'Tags', 'Tags AAT URL', 'Link Resource', 'Object Wikidata URL', 'Tags Wikidata URL'], axis=1)
    df = df.dropna(subset=["Department"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X1 = df.drop(columns=["Department"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y1 = df.loc[:, "Department"]
    X, _, y, _ = train_test_split(X1, y1, train_size= 30000, stratify=y1, random_state=0)
    met_objects = CustomDataset(X, y)
    return met_objects


def fetch_public_procurement():
    df = pd.read_csv("/home/soda/apajanir/categorical_embedding_with_LLMs/datasets/public_procurement.csv", delimiter=",")
    df = df.drop(['AWARD_EST_VALUE_EURO', 'AWARD_VALUE_EURO_FIN_1', 'TED_NOTICE_URL', 'VALUE_EURO', 'VALUE_EURO_FIN_1', 'VALUE_EURO_FIN_2'], axis=1)
    df = df.dropna(subset=["AWARD_VALUE_EURO"])
    # Supprimer la colonne target du DataFrame pour obtenir les features
    X1 = df.drop(columns=["AWARD_VALUE_EURO"])
    # Sélectionner uniquement la colonne target pour obtenir y
    y1 = df.loc[:, "AWARD_VALUE_EURO"]
    X, _, y, _ = train_test_split(X1, y1, train_size= 30000, random_state=0)
    public_procurement = CustomDataset(X, y)
    return public_procurement


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
