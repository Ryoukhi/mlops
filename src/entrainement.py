# entrainement.py
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ------------------------------------------------------------
# 1. Chemins
# ------------------------------------------------------------
CLEAN_PATH = 'E:\\EADL 5\\Gestion des déploiements des pipelines MLOps\\mlops\\data\\iris_clean.csv'
MODEL_DIR = 'E:\\EADL 5\\Gestion des déploiements des pipelines MLOps\\mlops\\models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.joblib')

# ------------------------------------------------------------
# 2. Chargement
# ------------------------------------------------------------
def charger_donnees_propres(chemin=CLEAN_PATH):
    df = pd.read_csv(chemin)
    print(" Dataset propre chargé :", df.shape)
    return df

# ------------------------------------------------------------
# 3. Préparation X / y
# ------------------------------------------------------------
def preparer_donnees(df, target='variety'):
    X = df.drop(columns=[target])
    y = df[target]
    print(f"   Features : {list(X.columns)}")
    print(f"   Classes  : {sorted(y.unique())}")
    return X, y

# ------------------------------------------------------------
# 4. Définition des 3 modèles
# ------------------------------------------------------------
def definir_modeles():
    """Retourne un dict {nom: pipeline}."""
    modeles = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
    }
    return modeles

# ------------------------------------------------------------
# 5. Entraînement + évaluation
# ------------------------------------------------------------
def entrainer_et_evaluer(X_train, X_test, y_train, y_test, modeles):
    resultats = {}

    print("\n" + "=" * 60)
    print("       ENTRAÎNEMENT ET ÉVALUATION DES 3 MODÈLES")
    print("=" * 60)

    for nom, pipeline in modeles.items():
        print(f"\n🔹 Modèle : {nom}")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Validation croisée (5 folds) sur tout le dataset
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

        resultats[nom] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
        }

        print(f"   Accuracy (test) : {acc:.4f}")
        print(f"   CV (5 folds)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("   Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("   Rapport :")
        print(classification_report(y_test, y_pred))

    return resultats

# ------------------------------------------------------------
# 6. Sélection du meilleur modèle
# ------------------------------------------------------------
def selectionner_meilleur(resultats):
    meilleur_nom = max(resultats, key=lambda n: resultats[n]["accuracy"])
    meilleur = resultats[meilleur_nom]
    print("\n" + "=" * 60)
    print(f"🏆 Meilleur modèle : {meilleur_nom}  (accuracy = {meilleur['accuracy']:.4f})")
    print("=" * 60)
    return meilleur_nom, meilleur["pipeline"]

# ------------------------------------------------------------
# 7. Sauvegarde
# ------------------------------------------------------------
def sauvegarder_modele(pipeline, chemin=BEST_MODEL_PATH):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    joblib.dump(pipeline, chemin)
    print(f"\n💾 Meilleur modèle sauvegardé : {chemin}")

# ------------------------------------------------------------
# 8. Pipeline principal
# ------------------------------------------------------------
def pipeline_entrainement():
    df = charger_donnees_propres()
    X, y = preparer_donnees(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    modeles = definir_modeles()
    resultats = entrainer_et_evaluer(X_train, X_test, y_train, y_test, modeles)

    nom, pipeline = selectionner_meilleur(resultats)
    sauvegarder_modele(pipeline)

    return resultats

if __name__ == "__main__":
    pipeline_entrainement()