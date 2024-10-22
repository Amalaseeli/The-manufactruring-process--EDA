import pandas as pd

def load_data():
    df=pd.read_csv("../failure_data.csv")
    return df

class DataFrameInfo:
    def __init__(self, df):
        self.df=df
    
    def describe_columns(self):
        print("Data Frame column descriptions")
        print(self.df.info())

    def extract_statistics(self):
        print("Statistical Values (Median, Standard Deviation, Mean):")
        self.df.describe()
        
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

    
if __name__=="__main__":
    df=load_data()
    data_info=DataFrameInfo(df)
    data_info.describe_columns()
    data_info.extract_statistics()
    data_info.count_distinct_values()
    data_info.dataframe_shape()
    data_info.null_values_info()

   

    
    
    