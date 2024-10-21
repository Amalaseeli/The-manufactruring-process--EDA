from sqlalchemy import create_engine

# Database connection details
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = "my-first-rds-db.cdyukqcc8a37.us-east-1.rds.amazonaws.com"
USER = 'postgres'
PASSWORD = "London8692!"
PORT = 5432
DATABASE = 'postgres'
engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")

engine.connect()

from sklearn.datasets import load_iris
import pandas as pd
data = load_iris()
iris = pd.DataFrame(data['data'], columns=data['feature_names'])
iris.head()

iris.to_sql('iris_dataset', engine, if_exists='replace')
df = pd.read_sql_table('iris_dataset', engine)
print(df.head())