import pandas as pd
import numpy as np
import os
import joblib

# ------------------------------------------------------------
# 1. Chemins
# ------------------------------------------------------------
BEST_MODEL_PATH = 'E:\\EADL 5\\Gestion des déploiements des pipelines MLOps\\mlops\\models\\best_model.joblib'
CLEAN_PATH = 'E:\\EADL 5\\Gestion des déploiements des pipelines MLOps\\mlops\\data\\iris_clean.csv'

# ------------------------------------------------------------
# 2. Chargement du modèle
# ------------------------------------------------------------
def charger_modele(chemin=BEST_MODEL_PATH):
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f" Modèle introuvable : {chemin}\n"
            "   → Lance d'abord entrainement.py"
        )
    modele = joblib.load(chemin)
    print(f" Modèle chargé depuis : {chemin}")
    return modele

# ------------------------------------------------------------
# 3. Prédiction sur nouvelles données
# ------------------------------------------------------------
def predire(modele, nouvelles_donnees):
    """
    nouvelles_donnees : DataFrame ou liste de dicts avec les 4 features.
    """
    if isinstance(nouvelles_donnees, list):
        nouvelles_donnees = pd.DataFrame(nouvelles_donnees)

    predictions = modele.predict(nouvelles_donnees)

    # Probabilités si le modèle les expose
    try:
        probas = modele.predict_proba(nouvelles_donnees)
        return predictions, probas
    except AttributeError:
        return predictions, None

# ------------------------------------------------------------
# 4. Test sur le dataset complet (évaluation finale)
# ------------------------------------------------------------
def tester_sur_dataset(modele, chemin=CLEAN_PATH):
    df = pd.read_csv(chemin)
    X = df.drop(columns=['variety'])
    y_vrai = df['variety']

    y_pred = modele.predict(X)
    acc = (y_pred == y_vrai).mean()

    print("\n" + "=" * 60)
    print("       ÉVALUATION FINALE SUR LE DATASET COMPLET")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}  ({int(acc * len(y_vrai))}/{len(y_vrai)})")

    print("\nAperçu des prédictions :")
    apercu = df.copy()
    apercu['prediction'] = y_pred
    apercu['correct'] = apercu['prediction'] == apercu['variety']
    print(apercu.head(10))

    return acc

# ------------------------------------------------------------
# 5. Test sur exemples manuels
# ------------------------------------------------------------
def tester_exemples(modele):
    exemples = [
        # [sepal_length, sepal_width, petal_length, petal_width]
        [5.1, 3.5, 1.4, 0.2],   # devrait être Setosa
        [6.0, 2.7, 5.1, 1.6],   # devrait être Versicolor
        [6.3, 3.3, 6.0, 2.5],   # devrait être Virginica
        [4.9, 3.0, 1.4, 0.2],   # devrait être Setosa
        [7.7, 3.8, 6.7, 2.2],   # devrait être Virginica
    ]
    colonnes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df_ex = pd.DataFrame(exemples, columns=colonnes)

    predictions, probas = predire(modele, df_ex)

    print("\n" + "=" * 60)
    print("       PRÉDICTIONS SUR EXEMPLES MANUELS")
    print("=" * 60)
    for i, (_, ligne) in enumerate(df_ex.iterrows()):
        print(f"\nExemple {i+1} : {ligne.tolist()}")
        print(f"   → Prédiction : {predictions[i]}")
        if probas is not None:
            proba_str = ", ".join(
                f"{cls}: {p:.3f}" for cls, p in zip(modele.classes_, probas[i])
            )
            print(f"   → Probabilités : {proba_str}")

# ------------------------------------------------------------
# 6. Pipeline principal
# ------------------------------------------------------------
def pipeline_test():
    modele = charger_modele()
    tester_sur_dataset(modele)
    tester_exemples(modele)

if __name__ == "__main__":
    pipeline_test()