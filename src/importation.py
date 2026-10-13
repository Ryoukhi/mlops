import pandas as pd

def charger_donnees(chemin):
    """Charge le fichier CSV dans un DataFrame pandas."""
    df = pd.read_csv(chemin)
    print("Données chargées :", df.shape)
    return df

file = charger_donnees('E:\EADL 5\Gestion des déploiements des pipelines MLOps\mlops\data\iris.csv')

