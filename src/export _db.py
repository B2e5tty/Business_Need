import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv(dotenv_path='src/.env')

# Fetch database connection parameters from environment variables
host='localhost',
port=5432,
database='telecom',
user='postgres',
password='1089elda'

def export_data_to_postgres(df):
    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

    # Export the DataFrame to PostgreSQL
    try:
        # Assuming user_satisfaction is your DataFrame
        df.to_sql('user_satisfaction', con=engine, if_exists='replace', index=False)

        # Verify the data by executing a query
        query = text("SELECT * FROM user_satisfaction")
        with engine.connect() as conn:
            result = conn.execute(query)

            # Convert the result to a Pandas DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            print(df.head())  # Display the first few rows of the DataFrame

    except Exception as e:
        print(f"An error occurred: {e}")
