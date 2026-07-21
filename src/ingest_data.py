import sqlite3
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PATH_DIR = Path(__file__).resolve().parent.parent / 'data' 


def sql_data():
    
    data_dir = PATH_DIR / 'Telco-Customer-Churn.csv'
    db_dir = PATH_DIR / 'churn_database.db'

    if not data_dir.exists():
        logger.error(f'No se encontró el archivo {data_dir}')
        raise FileNotFoundError(data_dir)
    
    logger.info(f'Leyendo CSV desde {data_dir}')
    df = pd.read_csv(data_dir)
    df.columns = df.columns.str.lower().str.replace('-', '_').str.replace(' ', '_')
    logger.info(f'{len(df)} filas, {len(df.columns)} columnas leídas')

    logger.info(f"Escribiendo tabla 'customer_churn' en {db_dir}")
    with sqlite3.connect(db_dir) as connection:
        df.to_sql('customer_churn', con=connection, if_exists='replace', index=False)
    logger.info('Carga a SQLite completada correctamente')

if __name__ == '__main__':
    sql_data()

