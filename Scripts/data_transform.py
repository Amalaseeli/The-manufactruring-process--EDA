import numpy as np
import pandas as pd

class DataTransform:
    def __init__(self,df:pd.DataFrame):
        self.df=df

    def drop_unwanted_columns(self,columns):
        self.df.drop(columns, inplace=True, axis=1)
    
    def convert_numeric_columns(self, columns:list):
        for col in columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def convert_to_category(self, columns):
        for col in columns:
            self.df[col]=self.df[col].astype('category') 

    def convert_binary_columns_to_boolean(self, columns):
        for col in columns:
            self.df[col] = self.df[col].astype(bool)  
    
    def transform_data(self):
        self.drop_unwanted_columns(['Unnamed: 0'])
        self.convert_numeric_columns(['Air temperature [K]','Process temperature [K]','Torque [Nm]', 'Tool wear [min]'])
        self.convert_to_category(['Type'])
        self.convert_binary_columns_to_boolean(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

if __name__ == "__main__":
    df=pd.read_csv('../failure_data.csv')
    # print(df.info())
    data_transform=DataTransform(df)
    data_transform.transform_data()
    
    print(df.head(5))
    print(df.info())
    print()