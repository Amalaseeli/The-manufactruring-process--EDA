import psycopg2
from sqlalchemy import create_engine

# Database connection details
host = "YOUR IDENTIFIER"
port = "5432"
database = "postgres"
user = "postgres"
password = "YOUR PASSWORD HERE!"

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