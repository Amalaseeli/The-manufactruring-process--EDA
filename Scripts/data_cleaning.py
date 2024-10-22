import pandas as pd
from data_information import DataFrameInfo
 
class Plotter:
    def __init__(self,df):
        self.df=df

class DataFrameTransform:
    def __init__(self, df):
        self.df=df
    
    def drop_null_columns(self):
        #print(self.df.isna().sum())
        DataFrameInfo.null_values_info(self)
      

if __name__=='__main__':
    df=pd.read_csv('../failure_data.csv')
    dataframetransform=DataFrameTransform(df)
    dataframetransform.drop_null_columns()
    
      