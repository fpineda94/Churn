import logging
import sys
from pathlib import Path

#sklearn modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline   


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from src.data_pipeline import get_data
from src.tracker import tracking_mlflow

RANDOM_STATE = 101
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def pipe(preprocessor):

    pipe_rf = Pipeline([('preprocessor', preprocessor), 
                        ('clf', RandomForestClassifier(random_state=RANDOM_STATE))

                        ])

    param_grid_rf = {
                    "clf__n_estimators": [100, 200, 400],
                    "clf__max_depth": [None, 5, 10, 20],
                    "clf__min_samples_leaf": [1, 5, 10],
                    "clf__class_weight": [None, "balanced"]
                    }
    return pipe_rf, param_grid_rf


def train_rf():

    try:
        logger.info('Cargadno y preparando datos desde el data pipeline')
        X_train, X_test, y_train, y_test, preprocessor = get_data()

    except Exception as e:
        logger.critical(f'Error crítico al cargar los datos desde el pipeline {e}')
        raise 

    model_rf, param_grid_rf = pipe(preprocessor)

    logger.info(f'Entrenando el modelo {model_rf} con GridSearchCV...')
    grid_model = GridSearchCV(model_rf,param_grid=param_grid_rf, cv = 5, n_jobs=-1, scoring='average_precision')
    grid_model.fit(X_train, y_train)
    logger.info(f'Mejor score CV: {grid_model.best_score_: .4f}')
    
    tracking_mlflow(grid_model.best_estimator_, X_test, y_test, run_name='random_forest', params=grid_model.best_params_)
    

    
if __name__ == '__main__':
    train_rf()