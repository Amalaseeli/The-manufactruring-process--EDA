import pandas as pd
from scipy import stats
from scipy.stats import yeojohnson
from Dataframeinfo import DataFrameInfo
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import numpy as np
 
class DataFrameTransform:
    def __init__(self, df):
        self.df=df

    
    def drop_null_columns(self):
        #print(self.df.isna().sum())
        null_info=DataFrameInfo.null_values_info(self)
        columns_to_drop=null_info[null_info['Percentage'] > 50]['Column']
        self.df = self.df.drop(columns=columns_to_drop)
        return null_info

    def impute_columns(self, impute_strategy={'Air temperature [K]':'mean',
                                              'Process temperature [K]':'mean',
                                              'Tool wear [min]':'median'}):
        
        for col, strategy in impute_strategy.items():
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    impute_value=self.df[col].mean()
                elif strategy == "median":
                    impute_value = self.df[col].median()
                else:
                    raise ValueError("strategy must be mean or median")
                self.df[col].fillna(impute_value, inplace=True)
                print(self.df.head(5))
                
        return self.df


    def identify_skew_columns(self):
        skewed_columns=[]
        for col in self.df.select_dtypes(include=['number']).columns:  # Only select numeric columns:
            print(f"{col}:", self.df[col].skew())
            if df[col].skew() < -0.5 or df[col].skew() > 0.5:
                skewed_columns.append(col)
        return skewed_columns

    def find_best_transformation(self, skewed_column):
        best_transformations = {}
        #log_transform
        for col in skewed_column:
            original_data=self.df[col]
            original_skew=self.df[col].skew()
             # Plot original data
            plt.figure(figsize=(12, 6))
            plt.title(f'Original Distribution of {col}')
            sns.histplot(original_data, bins=30, color='blue', kde=True, label=f'Skewness: {original_skew:.2f}')
            plt.legend()
            plt.savefig(f'../results/transformation/{col}_original.png')
            plt.close()

            log_trans = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)
             # Plot log-transformed data
            plt.figure(figsize=(12, 6))
            plt.title(f'Log Transformation of {col}')
            sns.histplot(log_trans, bins=30, color='green', kde=True, label=f'Skewness: {log_trans.skew():.2f}')
            plt.legend()
            plt.savefig(f'../results/transformation/{col}_log.png')
            plt.close()
            log_skew = log_trans.skew()

            if (self.df[col] > 0).all():
                boxcox_transform = stats.boxcox(self.df[col])
                boxcox_transform = pd.Series(boxcox_transform[0])
                # Plot Box-Cox transformed data
                plt.figure(figsize=(12, 6))
                plt.title(f'Box-Cox Transformation of {col}')
                sns.histplot( boxcox_transform , bins=30, color='purple', kde=True, label=f'Skewness: { boxcox_transform .skew():.2f}')
                plt.legend()
                plt.savefig(f'../results/transformation/{col}_boxcox.png')
                plt.close()
                boxcox_skew= boxcox_transform.skew()

            yeojohnson_transform = stats.yeojohnson(self.df[col])
            yeojohnson_transform= pd.Series(yeojohnson_transform[0])
            yeojohnson_skew=yeojohnson_transform.skew()
            # Plot Yeo-Johnson transformed data
            plt.figure(figsize=(12, 6))
            plt.title(f'Yeo-Johnson Transformation of {col}')
            sns.histplot(yeojohnson_transform, bins=30, color='red', kde=True, label=f'Skewness: {yeojohnson_transform.skew():.2f}')
            plt.legend()
            plt.savefig(f'../results/transformation/{col}_yeojohnson.png')
            plt.close()
            
            print(f"log_skew:{log_skew},boxcox_skew:{boxcox_skew},yeojohnson_skew:{yeojohnson_skew} ")

            skews = {
                'original': original_skew,
                'log': log_skew,
                'boxcox': boxcox_skew,
                'yeojohnson': yeojohnson_skew
            }
            best_transformation = min(skews, key=lambda k: abs(skews[k]))
            best_skew_value = skews[best_transformation]
            print(f"best_transformation:{ best_transformation },best_skew_value:{best_skew_value } ")

            best_transformations[col] = {
                'best_transformation': best_transformation,
                'best_skew_value': best_skew_value,
                'all_skew_values': skews
            }
            print(best_transformations)

              # Applying the best transformation to the dataframe
            if best_transformation == 'log':
                self.df[col] = log_trans
            elif best_transformation == 'boxcox':
                self.df[col] = boxcox_transform
            elif best_transformation == 'yeojohnson':
                self.df[col] = yeojohnson_transform
            return best_transformations
        
class Plotter:
    def __init__(self, df):
        self.df=df
    
    def plot_nulls(self, nulls_before, nulls_after):
        plt.figure(figsize=(12,8))
        sns.barplot(x=nulls_before.index, y=nulls_before['Null Count'], color='blue', alpha=0.6, label='Before')
        sns.barplot(x=nulls_after.index, y=nulls_after['Null Count'], color='red', alpha=0.6, label='After')
        plt.title('NULL Values Before and After Imputation')
        plt.ylabel('Number of NULL Values')
        plt.savefig('../results/null_values.png')
        plt.show()
    
    def hist_plot_for_skewed_columns(self):
        plt.figure(figsize=(16,10))
        self.df.hist(bins=50 )
        plt.subplots_adjust(hspace=0.8, wspace=0.7)
        plt.tight_layout()
        plt.savefig('../results/skew/histogram/hist_skew.png')
        plt.show()
    
    def box_plot_for_skewed_columns(self):
        #numeric_columns=['UDI','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]']
        for col in self.df.select_dtypes(include=['number']).columns:
            sns.boxplot(self.df[col])
            plt.title(col+'skew')
            plt.tight_layout()
            plt.savefig(f'../results/skew/box_plot/{col}_box_plot.png')
            plt.show()
    
    def Q_plot_for_skewed_columns(self):
        #numeric_columns=['UDI','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]']
        
        for col in self.df.select_dtypes(include=['number']).columns:
            print(self.df['Air temperature [K]'].isnull().sum())
            qq_plot = qqplot(self.df[col] ,line='q', fit=True)
            plt.title(col)
            plt.savefig(f'../results/skew/Q_plot/{col}_Q_plot.png')
            pyplot.show()

    def box_plot_for_outlier(self):
        for col in self.df.select_dtypes(include=['number']).columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(self.df[col], color='lightgreen', showfliers=True)
            sns.swarmplot(self.df[col], color='black', size=3)
            plt.title(f'Box plot with scatter points of {col}')
            plt.tight_layout()
            plt.savefig(f'../results/outlier/box_plot/{col}_box_plot.png')
            plt.show()

    def plot_scatter_plot(self, target_column):
        # for column in df.select_dtypes(include=['number']).columns:
        for column in self.df.select_dtypes(include=['number']).columns:
            if column != target_column:  # Skip the target column itself
                plt.figure(figsize=(8, 6))
                plt.scatter(df[column], df[target_column], alpha=0.7)
                plt.title(f'Scatter Plot of {column} vs {target_column}')
                plt.xlabel(column)
                plt.ylabel(target_column)
                plt.grid(True)
                plt.show()
                 
if __name__=='__main__':
    df=pd.read_csv('../Dataset/cleaned_failure_data.csv')
    target_column=df['Machine failure']
    dataframetransform=DataFrameTransform(df)
    nulls_before = dataframetransform.drop_null_columns()
    # Impute missing values
    df=dataframetransform.impute_columns()
    
    #Get null values after imputation
    nulls_after = dataframetransform.drop_null_columns()
    
    plotter=Plotter(df)
    #plotter.plot_nulls( nulls_before ,nulls_after)
    skewed_columns= dataframetransform.identify_skew_columns()
    print(f"skewed_columns:", skewed_columns)
    #plotter.hist_plot_for_skewed_columns()
    #plotter.box_plot_for_skewed_columns()
    #plotter.Q_plot_for_skewed_columns()
    #transformations_info=dataframetransform.find_best_transformation(skewed_columns)
    plotter.plot_scatter_plot(target_column)
    #plotter.box_plot_for_outlier()
   
        