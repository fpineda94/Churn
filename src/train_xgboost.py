import logging
import sys
from pathlib import Path

#sklearn modules
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline   
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_pipeline import get_data
from src.tracker import tracking_mlflow

RANDOM_STATE = 101
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def pipe(preprocessor, y_train):

    pipe_xgb = Pipeline([('preprocessor', preprocessor),
                        ('clf', xgb.XGBClassifier(random_state = RANDOM_STATE, 
                                                  eval_metric = 'logloss'))
                        ])

    param_grid_xgb = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample": [0.7, 0.9, 1.0],
    "clf__scale_pos_weight": [1, (y_train==0).sum()/(y_train==1).sum()]  # maneja desbalance
}
    return pipe_xgb, param_grid_xgb


def train_xgb():
    try:
        logger.info('Cargadno y preparando datos desde el data pipeline')
        X_train, X_test, y_train, y_test, preprocessor = get_data()

    except Exception as e:
        logger.critical(f'Error crítico al cargar los datos desde el pipeline {e}')
        raise 


    model_xgb, param_grid_xgb = pipe(preprocessor, y_train)

    logger.info(f'Entrenando el modelo {model_xgb} con GridSearchCV...')
    grid_model = GridSearchCV(model_xgb,param_grid=param_grid_xgb, cv = 5, n_jobs=-1, scoring='average_precision')
    grid_model.fit(X_train, y_train)
    logger.info(f'Mejor score CV: {grid_model.best_score_: .4f}')
    
    tracking_mlflow(grid_model.best_estimator_, X_test, y_test, run_name='xgboost', params=grid_model.best_params_)

if __name__ == '__main__':
    train_xgb()