# kubeflow_pipeline.py
from pathlib import Path

from kfp import compiler, dsl
from kfp.kubernetes import mount_pvc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINES_DIR = PROJECT_ROOT / 'pipelines'
DEFAULT_DATA_PATH = '/data/iris.csv'


def attach_iris_pvc(task):
    return mount_pvc(
        task,
        pvc_name='iris-data-pvc',
        mount_path='/data',
    )


@dsl.component(
    base_image='python:3.11-slim',
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'joblib']
)
def charger_donnees(input_csv: str, output_dataset: dsl.Output[dsl.Dataset]):
    import pandas as pd

    df = pd.read_csv(input_csv)
    df.to_csv(output_dataset.path, index=False)
    print(f'✅ Données chargées : {df.shape}')


@dsl.component(
    base_image='python:3.11-slim',
    packages_to_install=['pandas', 'numpy']
)
def nettoyer_donnees(input_dataset: dsl.Input[dsl.Dataset], output_dataset: dsl.Output[dsl.Dataset]):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(input_dataset.path)
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    df.columns = [str(c).strip().lower().replace('.', '_') for c in df.columns]
    df.to_csv(output_dataset.path, index=False)
    print(f'✅ Nettoyage terminé : {df.shape}')


@dsl.component(
    base_image='python:3.11-slim',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def entrainer_modeles(input_dataset: dsl.Input[dsl.Dataset], output_model: dsl.Output[dsl.Model]):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(input_dataset.path)
    X = df.drop(columns=['variety'])
    y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modeles = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    meilleur = None
    meilleure_acc = -1
    meilleur_pipeline = None

    for nom, pipeline in modeles.items():
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        print(f'🧪 {nom} : {acc:.4f}')
        if acc > meilleure_acc:
            meilleure_acc = acc
            meilleur = nom
            meilleur_pipeline = pipeline

    print(f'🏆 Meilleur modèle : {meilleur} ({meilleure_acc:.4f})')
    joblib.dump(meilleur_pipeline, output_model.path)


@dsl.component(
    base_image='python:3.11-slim',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def tester_modele(input_dataset: dsl.Input[dsl.Dataset], input_model: dsl.Input[dsl.Model]):
    import joblib
    import pandas as pd

    df = pd.read_csv(input_dataset.path)
    X = df.drop(columns=['variety'])
    y_vrai = df['variety']

    modele = joblib.load(input_model.path)
    y_pred = modele.predict(X)
    acc = (y_pred == y_vrai).mean()
    print(f'✅ Accuracy finale : {acc:.4f}')


@dsl.pipeline(
    name='iris-classification-pipeline',
    description='Pipeline complet de data prep, entraînement et validation pour Iris'
)
def iris_pipeline(input_csv: str = DEFAULT_DATA_PATH):
    etape_chargement = attach_iris_pvc(charger_donnees(input_csv=input_csv))
    etape_nettoyage = attach_iris_pvc(
        nettoyer_donnees(input_dataset=etape_chargement.outputs['output_dataset'])
    )
    etape_entrainement = attach_iris_pvc(
        entrainer_modeles(input_dataset=etape_nettoyage.outputs['output_dataset'])
    )
    tester_modele(
        input_dataset=etape_nettoyage.outputs['output_dataset'],
        input_model=etape_entrainement.outputs['output_model']
    )


if __name__ == '__main__':
    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PIPELINES_DIR / 'iris_pipeline.yaml'
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=str(output_file)
    )
    print(f'✅ Compilé → {output_file}')