import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def load_data(csv_path: str, table_name: str):
    load_dotenv()

    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    df = pd.read_csv(csv_path)
    print(f"{len(df)} rows, {len(df.columns)} columns")

    db_url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df.to_sql(
            table_name,
            con=conn,
            if_exists="replace",
            index=False,
            chunksize=500,
            method="multi"
        )

load_data("data/arm_in_move.csv", "arm_in_move")
load_data("data/arm_no_move.csv", "arm_no_move")
load_data("data/chest_in_move.csv", "chest_in_move")
load_data("data/chest_no_move.csv", "chest_no_move")
