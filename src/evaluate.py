import logging
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR}/mlflow.db"
EXPERIMENT_NAME = "churn_prediction"
METRIC_TO_OPTIMIZE = "metrics.avg_precision"


def evaluate_and_register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        logger.error(f"No se encontró el experimento '{EXPERIMENT_NAME}'")
        return

    logger.info(f"Buscando el mejor modelo en el experimento '{EXPERIMENT_NAME}' ordenado por '{METRIC_TO_OPTIMIZE}'...")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"{METRIC_TO_OPTIMIZE} DESC"]
    )

    if not runs:
        logger.warning("No se encontraron corridas registradas en MLflow.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_run_name = best_run.data.tags.get("mlflow.runName", "Desconocido")
    best_score = best_run.data.metrics.get("avg_precision", 0.0)

    logger.info("=" * 50)
    logger.info(f"MODELO GANADOR SELECCIONADO: {best_run_name}")
    logger.info(f"Run ID: {best_run_id}")
    logger.info(f"Best PR-AUC (avg_precision): {best_score:.4f}")
    logger.info("=" * 50)

    model_uri = f"runs:/{best_run_id}/model"
    model_name = "Churn_Production_Model"

    logger.info(f"Registrando el modelo en el Model Registry como '{model_name}'...")
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=registered_model.version
    )

    logger.info(f"El modelo versión {registered_model.version} ha sido marcado con el alias 'Champion'.")


if __name__ == '__main__':
    evaluate_and_register_best_model()