# Projet MLOps Iris avec Kubeflow et MLflow

Ce projet est un mini pipeline MLOps complet pour le dataset Iris :
- chargement des données
- nettoyage
- entraînement de plusieurs modèles
- sélection du meilleur modèle
- export du pipeline en YAML pour Kubeflow
- suivi des runs avec MLflow
- préparation du dataset pour l’import dans l’interface Kubeflow

## Structure du projet

```text
mlops/
├── data/
│   ├── iris.csv
│   └── iris_clean.csv
├── kind/
│   ├── kind-config.yaml
│   ├── kubeflow-install.sh
│   ├── mount-iris-data.sh
│   ├── mount-iris-data-pod.sh
│   ├── iris-data-pvc.yaml
│   └── iris-data-mount.yaml
├── models/
│   └── best_model.joblib
├── pipelines/
│   └── iris_pipeline.yaml
├── src/
│   ├── entrainement.py
│   ├── kubeflow_pipeline.py
│   ├── nettoyage.py
│   ├── importation.py
│   ├── mlflow_tracking.py
│   └── test.py
├── README.md
├── requirements.txt
└── .venv/
```

## Prérequis

Assure-toi d’avoir installé :
- Docker
- Kind
- kubectl
- Python 3.10+
- pip

## 1) Créer le cluster Kind

```bash
cd /chemin/vers/mlops
kind create cluster --name mlops-kind --config kind/kind-config.yaml
```

## 2) Vérifier le cluster

```bash
kubectl get nodes
kubectl get ns
```

## 3) Vérifier Kubeflow

Si Kubeflow est déjà installé :

```bash
kubectl get pods -A | grep -i kubeflow
kubectl get ns
```

Pour accéder à l’UI :

```bash
kubectl port-forward -n kubeflow svc/centraldashboard 8081:80
```

Puis ouvrez :

```text
http://localhost:8081
```

## 4) Monter le dataset Iris dans le cluster

Le fichier source est :

```text
data/iris.csv
```

### Option A : script de montage simple

```bash
chmod +x kind/mount-iris-data.sh
./kind/mount-iris-data.sh
```

Ce script crée un pod de test et monte le dataset dans :

```text
/data/iris.csv
```

### Option B : pod avec PVC dédié

```bash
kubectl apply -f kind/iris-data-pvc.yaml
bash ./kind/mount-iris-data-pod.sh
```

Puis vérifie :

```bash
kubectl exec -n kubeflow iris-data-loader -- ls -l /data
kubectl exec -n kubeflow iris-data-loader -- head -n 5 /data/iris.csv
```

Important : dans un pipeline Kubeflow réel, le bon mécanisme est un PVC monté dans chaque composant, car chaque composant est un conteneur séparé. C’est ce qui est préparé dans `src/kubeflow_pipeline.py`.

## 5) Générer le fichier YAML du pipeline Kubeflow

Depuis la racine du projet :

```bash
cd /chemin/vers/mlops
python src/kubeflow_pipeline.py
```

Le fichier généré est :

```text
pipelines/iris_pipeline.yaml
```

## 6) Importer le pipeline dans l’UI Kubeflow

Dans l’interface Kubeflow :
1. clique sur “Upload pipeline”
2. sélectionne `pipelines/iris_pipeline.yaml`
3. importe le pipeline
4. lance le pipeline

La valeur par défaut du paramètre d’entrée est déjà prête pour :

```text
/data/iris.csv
```

## 7) Lancer le projet localement

Pour entraîner le modèle sans passer par Kubeflow :

```bash
cd /chemin/vers/mlops
python src/entrainement.py
```

Le meilleur modèle est sauvegardé dans :

```text
models/best_model.joblib
```

## 8) Démarrer MLflow

Installation si nécessaire :

```bash
python -m pip install mlflow
```

Lancer l’UI :

```bash
cd /chemin/vers/mlops
mlflow ui --host 0.0.0.0 --port 5000
```

Puis ouvre :

```text
http://localhost:5000
```

Le tracking est configuré dans `src/mlflow_tracking.py`.

## 9) Flux normal du projet

```text
dataset iris.csv
   ↓
PVC / montage dans Kubeflow
   ↓
pipeline Kubeflow
   ↓
chargement + nettoyage + entraînement + validation
   ↓
modèle enregistré
   ↓
tracking MLflow
```

## 10) Commandes utiles

```bash
# Créer le cluster Kind
kind create cluster --name mlops-kind --config kind/kind-config.yaml

# Vérifier Kubeflow
kubectl get pods -A | grep -i kubeflow

# Accéder à Kubeflow UI
kubectl port-forward -n kubeflow svc/centraldashboard 8081:80

# Monter le dataset dans le cluster
./kind/mount-iris-data.sh

# Vérifier le montage
kubectl exec -n kubeflow iris-data-loader -- ls -l /data
kubectl exec -n kubeflow iris-data-loader -- head -n 5 /data/iris.csv

# Générer le YAML du pipeline
python src/kubeflow_pipeline.py

# Lancer le modèle local
python src/entrainement.py

# Lancer MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

## 11) Points clés

- Le bon chemin d’entrée pour le pipeline est : `/data/iris.csv`
- Le pipeline est exportable dans Kubeflow UI via `pipelines/iris_pipeline.yaml`
- Le dataset doit être accessible via un PVC ou un montage explicite dans le cluster
- MLflow permet de suivre les métriques comme `accuracy` et le modèle produit

## 12) Étape suivante possible

Tu peux étendre le projet avec :
- un service de prédiction FastAPI
- un dépôt de modèles MLflow plus avancé
- un déploiement Docker/Kubernetes du modèle
- une version plus complète avec versioning des données et pipelines en production
