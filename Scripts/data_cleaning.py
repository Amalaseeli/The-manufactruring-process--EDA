import pandas as pd

def load_data():
    df=pd.read_csv("../failure_data.csv")
    print(df.shape)
    print(df.head(5))
    return df



if __name__=="__main__":
    df=load_data()
    
    
    