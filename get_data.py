import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def get_data(table_name: str, columns: list[str] | None = None) -> np.ndarray:
    load_dotenv()

    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    db_url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    query = text(f"SELECT {', '.join(columns) if columns else '*'} FROM {table_name}")

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df.to_numpy()