import pandas as pd
from Datatransform import DataTransform

def load_data():
    df=pd.read_csv("../failure_data.csv")
    data_transform = DataTransform(df)
    data_transform.transform_data()
    return df

class DataFrameInfo:
    def __init__(self, df):
        self.df=df
    
    def describe_columns(self):
        print("Data Frame column descriptions")
        print(self.df.info())

    def extract_statistics(self):
        print("Statistical Values (Median, Standard Deviation, Mean):")
        print(self.df.describe())
        
    def count_distinct_values(self):
        """
        Count distinct values in each column.
        """
        print("Distinct Value Counts per Column:")
        categorical_columns = self.df.select_dtypes(include=['object', 'category'])
        distinct_counts = categorical_columns.nunique()
        print(distinct_counts)
    
    def dataframe_shape(self):
        print("DataFrame Shape:")
        shape= self.df.shape
        print(shape)
    
    def null_values_info(self):
        print("Null Values Information:")
        null_counts = self.df.isnull().sum()
        null_percentages = (self.df.isnull().mean() * 100).round(2)
        null_info = pd.DataFrame({'Null Count': null_counts, 'Percentage': null_percentages})
        print(null_info)
        return null_info
    
    def data_frame_information(self):
        self.describe_columns()
        self.extract_statistics()
        self.count_distinct_values()
        self.dataframe_shape()
        self.null_values_info()
        return self.df
    
if __name__=="__main__":
    df=load_data()
    data_info=DataFrameInfo(df)
    data_info.data_frame_information()
    

   

    
    
    