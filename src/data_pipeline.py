import pandas as pd
import sqlite3
from pathlib import Path
import logging

#sklearn modules
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


DB_PATH_DIR = Path(__file__).resolve().parent.parent / 'data' / 'churn_database.db'
logger = logging.getLogger(__name__)
RANDOM_STATE = 101


def load_data() -> pd.DataFrame:

    with sqlite3.connect(DB_PATH_DIR) as connection:
        query = "SELECT * FROM customer_churn"
        df = pd.read_sql(query, connection)
    logger.info(f"{len(df)} filas cargadas desde SQLite")
    return df 
    

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    
    if 'totalcharges' in df.columns:
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        df['totalcharges'] = df['totalcharges'].fillna(0.0)
        logger.info("Columna 'totalcharges' convertidaa númerico (float)")
    return df


def split_data(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        logger.info('Separando datos para entrenamiento')
        X = df.drop(['customerid', 'churn'], axis=1)
        y = df['churn'].map({'No': 0, 'Yes': 1})
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size= test_size, 
                                                            random_state=RANDOM_STATE,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test



def build_prepocessor(df: pd.DataFrame) -> ColumnTransformer:
     
    cols = df.drop(['customerid', 'churn'], errors= 'ignore', axis=1)
    num_cols = cols.select_dtypes(exclude = 'object').columns.tolist()
    cat_cols = cols.select_dtypes(include = 'object').columns.tolist()
    
    logger.info(f"Columnas categóticas {cat_cols}")
    logger.info(f"Columnas numéricas {num_cols}")

    preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop='first', sparse_output=False,  handle_unknown='ignore'), cat_cols)])

    return preprocessor

def get_data(test_size: float=0.2):
     df = load_data()
     df = preprocessing(df)
     preprocessor = build_prepocessor(df)

     X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)

     return X_train, X_test, y_train, y_test, preprocessor

if __name__ =='__main__':

    X_train, X_test, y_train, y_test, preprocessor = get_data()
    logger.info(f"Pipeline probado exitosamente. Train shape: {X_train.shape}, Test shape: {X_test.shape}")