import yaml
from sqlalchemy import create_engine
import pandas as pd

def load_crediential():
    with open('crediential.yaml', 'r') as f:
        credentials=yaml.safe_load(f)
    return credentials
    
        
class RDSDatabaseConnector():
    def __init__(self, credentials:dict):
         self.credentials = credentials

        
    def init_engine(self):
        RDS_HOST=self.credentials.get("RDS_HOST")#
        RDS_PASSWORD=self.credentials.get("RDS_PASSWORD")
        RDS_USER=self.credentials.get("RDS_USER")
        RDS_DATABASE=self.credentials.get("RDS_DATABASE")
        RDS_PORT=self.credentials.get("RDS_PORT")

        engine = create_engine(f"postgresql://{RDS_USER}:{ RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DATABASE}")
        engine.connect()
        return engine
    
    def fetch_data(self):
        """
        Extracts data from the failure_data table and returns it as a Pandas DataFrame.

        Returns:
            DataFrame: A Pandas DataFrame containing the data from the failure_data table.
        """
        # Initialize the engine
        engine = self.init_engine()
        
        # Query to extract data from the failure_data table
        query = "SELECT * FROM failure_data"
        
        # Execute the query and store the result in a DataFrame
        df = pd.read_sql(query, engine)
        
        return df
    
credentials=load_crediential()
connector = RDSDatabaseConnector(credentials)

# Fetch the data
data_frame = connector.fetch_data()

# Display the data
print(data_frame)
data_frame.to_csv('failure_data.csv')
