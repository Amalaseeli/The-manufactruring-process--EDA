import psycopg2
from sqlalchemy import create_engine

# Database connection details
host = "my-first-rds-db.cdyukqcc8a37.us-east-1.rds.amazonaws.com"
port = "5432"
database = "postgres"
user = "postgres"
password = "London8692!"

# Establish the connection
conn = psycopg2.connect(
    host=host,
    port=port,
    database=database,
    user=user,
    password=password
)

# Perform database operations using the connection
# ...

# Close the connection when finished
conn.close()