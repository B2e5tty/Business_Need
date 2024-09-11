# scripts/load_data.py

import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine,text


# Load environment variables from .env file
load_dotenv(dotenv_path='src/.env')

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the database
        connection = psycopg2.connect(
            host='localhost',
            port=5432,
            database='telecom',
            user='postgres',
            password='1089elda'
        )

        # Load data using pandas
        df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Export dataframe to database
def export_data_to_postgres(df):
    # Fetch database connection parameters from environment variables
    host='localhost'
    port=5432
    database='telecom'
    user='postgres'
    password='1089elda'

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


def load_data_using_sqlalchemy(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query using SQLAlchemy.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create a connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, engine)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None