import os
from pathlib import Path

try:
    import mlflow
    from mlflow.models import infer_signature
except ImportError:  # pragma: no cover
    mlflow = None
    infer_signature = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f'file://{PROJECT_ROOT / "mlruns"}')


def configure_tracking(tracking_uri=None):
    if mlflow is None:
        print('MLflow non installé, tracking désactivé.')
        return None

    uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI') or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('iris-classification')
    return uri


def log_training_run(pipeline, X_train, X_test, y_train, y_test, model_name='best_model', model_path=None):
    if mlflow is None:
        return None

    configure_tracking()
    with mlflow.start_run(run_name=model_name) as run:
        from sklearn.metrics import accuracy_score

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_metric('accuracy', float(accuracy))

        try:
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path='model',
                registered_model_name='iris-best-model',
                signature=signature,
            )
        except Exception:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path='model',
                registered_model_name='iris-best-model',
            )

        if model_path:
            mlflow.log_artifact(model_path, artifact_path='saved_model')

        print(f'MLflow run enregistré : {run.info.run_id}')
        return run.info.run_id
