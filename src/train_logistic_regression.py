import logging
import sys
from pathlib import Path

#sklearn modules
from sklearn.linear_model import LogisticRegression
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

    pipe = Pipeline([('preprocessor', preprocessor),
                 ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    
    param_grid = [
        # L2 (Ridge)
        {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__solver": ["lbfgs", "liblinear"], "clf__l1_ratio": [0]},
        # L1 (Lasso)
        {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__solver": ["liblinear", "saga"], "clf__l1_ratio": [1]},
        # ElasticNet
        {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__l1_ratio": [0.2, 0.5, 0.8], "clf__solver": ["saga"]}]

    return pipe, param_grid


def train():

    try:
        logger.info('Cargando y preparando datos desde el data pipeline')
        X_train, X_test, y_train, y_test, preprocessor = get_data()
    except Exception as e:
        logger.critical(f'Error crítico al cargar los datos desde el pipeline {e}')
        raise 

    model, param_grid = pipe(preprocessor=preprocessor)

    logger.info(f'Entrenando el modelo {model} con GridSearchCV...')
    grid_model = GridSearchCV(model,param_grid=param_grid, cv = 5, n_jobs=-1, scoring='average_precision')
    grid_model.fit(X_train, y_train)
    logger.info(f'Mejor score CV: {grid_model.best_score_: .4f}')

    tracking_mlflow(grid_model.best_estimator_, X_test, y_test, run_name='logreg', params=grid_model.best_params_)


if __name__ == '__main__':
    train()