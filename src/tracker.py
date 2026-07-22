import logging
import mlflow
import mlflow.sklearn
from typing import Tuple
import numpy as np

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, classification_report)

logger = logging.getLogger(__name__)


def calcular_metricas(modelo, X_test, y_test) -> Tuple[np.ndarray, np.ndarray, dict]:

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    reporte = classification_report(y_test, y_pred, output_dict=True)

    metricas =  {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "f1_churn": f1_score(y_test, y_pred, pos_label=1),
        "precision_churn": reporte["1"]["precision"],
        "recall_churn": reporte["1"]["recall"],
        "accuracy": reporte["accuracy"],
    }

    return y_pred, y_prob, metricas




def tracking_mlflow(modelo, X_test, y_test, run_name: str, params: dict,
                    experiment_name: str = 'churn_prediction') -> dict:

    mlflow.set_experiment(experiment_name)
    y_pred, y_prob, metricas = calcular_metricas(modelo, X_test, y_test)

    with mlflow.start_run(run_name=run_name):

        if params:
            mlflow.log_params(params)

        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, artifact_path='model')
        logger.info(f'Run {run_name} registrado')
        logger.info(f'Métricas registradas: {metricas}')

    return metricas
    