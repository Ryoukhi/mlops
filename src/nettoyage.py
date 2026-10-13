import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------
DATA_PATH = 'E:\\EADL 5\\Gestion des déploiements des pipelines MLOps\\mlops\\data\\iris.csv'
CLEAN_PATH = 'E:\\EADL 5\\Gestion des déploiements des pipelines MLOps\\mlops\\data\\iris_clean.csv'

def charger_donnees(chemin=DATA_PATH):
    """Charge le fichier CSV dans un DataFrame pandas."""
    df = pd.read_csv(chemin)
    print("Données chargées :", df.shape)
    return df

# ------------------------------------------------------------
# 2. Inspection des données
# ------------------------------------------------------------
def inspecter_donnees(df):
    """Affiche un rapport d'inspection sur le DataFrame."""
    print("\n===== APERÇU =====")
    print(df.head())

    print("\n===== TYPES =====")
    print(df.dtypes)

    print("\n===== VALEURS MANQUANTES =====")
    print(df.isnull().sum())

    print("\n===== DOUBLONS =====")
    print(f"Nombre de doublons : {df.duplicated().sum()}")

    print("\n===== STATISTIQUES DESCRIPTIVES =====")
    print(df.describe())

    print("\n===== RÉPARTITION DES CLASSES =====")
    print(df['variety'].value_counts())

# ------------------------------------------------------------
# 3. Nettoyage
# ------------------------------------------------------------
def nettoyer_donnees(df):
    """Nettoie le DataFrame : doublons, valeurs manquantes, types."""
    df = df.copy()

    # 3.1 Suppression des doublons
    avant = df.shape[0]
    df = df.drop_duplicates()
    print(f"\n🧹 Doublons supprimés : {avant - df.shape[0]}")

    # 3.2 Gestion des valeurs manquantes (numériques -> médiane)
    colonnes_num = df.select_dtypes(include=[np.number]).columns
    for col in colonnes_num:
        if df[col].isnull().sum() > 0:
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            print(f"   → {col} : valeurs manquantes remplacées par la médiane ({mediane})")

    # 3.3 Normalisation des noms de colonnes (optionnel)
    df.columns = [c.strip().lower().replace('.', '_') for c in df.columns]

    # 3.4 Vérification finale
    print("\n Nettoyage terminé.")
    print("   Shape finale :", df.shape)
    print("   Valeurs manquantes restantes :", df.isnull().sum().sum())

    return df

# ------------------------------------------------------------
# 4. Sauvegarde
# ------------------------------------------------------------
def sauvegarder(df, chemin=CLEAN_PATH):
    """Sauvegarde le DataFrame nettoyé en CSV."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    df.to_csv(chemin, index=False)
    print(f"\n Dataset nettoyé sauvegardé : {chemin}")

# ------------------------------------------------------------
# 5. Pipeline principal
# ------------------------------------------------------------
def pipeline_nettoyage():
    df = charger_donnees()
    inspecter_donnees(df)
    df_clean = nettoyer_donnees(df)
    sauvegarder(df_clean)
    return df_clean

if __name__ == "__main__":
    pipeline_nettoyage()