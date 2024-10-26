import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analyse_Visualize:
    def __init__(self,df):
        self.df=df

    def analyse_operating_ranges(self):
        columns=[
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
                ]
        operating_ranges= df[columns].agg(['min', 'max', 'mean', 'std']).transpose()
        operating_ranges.columns=['Minimum', 'Maximum', 'Mean', 'Standard Deviation']
        print(operating_ranges)

        #groping by quality type
        grouped_ranges = df.groupby('Type')[columns].agg(['min', 'max', 'mean', 'std']).transpose()
        print(grouped_ranges)

    def plot_tool_wear(self):
        plt.figure(figsize=(10,6))
        plt.hist(df['Tool wear [min]'], bins=20, color='skyblue', edgecolor="black")
        plt.xlabel('Tool wear [min]')
        plt.ylabel('Number of Tools')
        plt.title('Distribution of Tool Wear Values')
        plt.grid(axis='y',linestyle='--', alpha=0.8)
        #plot upperlimit
        max_tool_wear = df['Tool wear [min]'].max()
        plt.axvline(max_tool_wear, color='red', linestyle='dashed', linewidth=1.5, label=f'Max Tool Wear: {max_tool_wear:.2f} min')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../results//visualization/toolwear_hist.png')
        plt.show()

    def determining_causes_of_failure(self):
        # print(df[df['Machine failure']==True].sum())
        total_failures= df['Machine failure'].sum()
        failure_percentage=total_failures/len(df['Machine failure'])*100
        print(f"Total failure:", total_failures)
        print(f"Failure Percentage", failure_percentage)

        #Check if the failures are being caused based on the quality of the product.
        failures_by_quality = df.groupby('Type')['Machine failure'].sum()
        print(failures_by_quality)

        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF','RNF']
        failures_by_cause = df[failure_columns].sum()
        print(failures_by_cause)
        return failures_by_quality , failures_by_cause

    def plot_failure_nonfailure_distrubution(self,failures_by_quality, failures_by_cause):
        total_records=len(df)
        total_failures=df['Machine failure'].sum()
        non_failure=total_records- total_failures

        labels=["Failure", "Non_failure"]
        plt.pie([total_failures , non_failure], labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'skyblue'],startangle=140)
        plt.title('Percentage of Failures in the Manufacturing Process')
        plt.savefig('../results/visualization/failure_non_failure_pie.png')
        plt.show()

        #plotting bar chart for failures by product quality
        failures_by_quality.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(10, 6))
        plt.title('Number of Failures by Product Quality')
        plt.xlabel('Product Quality Type')
        plt.ylabel('Number of Failures')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/visualization/failures_by_product.png')
        plt.show()  

        failures_by_cause.plot(kind='bar', color='purple', edgecolor='black', figsize=(10, 6))
        plt.title('Number of Failures by Cause')
        plt.xlabel('Failure Cause')
        plt.ylabel('Number of Failures')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/visualization/failures_by_cause.png')
        plt.show()

    def understad_of_failure(self):
        machine_settings=  ['Torque [Nm]', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]']
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF','RNF']
        # Analyze correlation between machine settings and failures
        correlation_results={}
        for failure in failure_types:
            correlations=df[machine_settings + [failure]].corr()[failure].drop(failure)
            #print(correlations)
            correlation_results[failure] = correlations

        # Display correlation results
        for failure, corr_values in correlation_results.items():
            print(f"Correlation between machine settings and {failure}:")
            print(corr_values, "\n")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Torque [Nm]', y='Air temperature [K]', hue='TWF', data=df, alpha=0.6)
        plt.title('Scatter Plot of Torque vs. Air Temperature by TWF (Tool Wear Failure)')
        plt.xlabel('Torque [Nm]')
        plt.ylabel('Air temperature [K]')
        plt.legend(title='TWF')
        plt.grid(True)
        plt.savefig('../results/visualization/scatter_plot.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='TWF', y='Rotational speed [rpm]', data=df)
        plt.title('Box Plot of Rotational Speed by TWF (Tool Wear Failure)')
        plt.xlabel('TWF (Tool Wear Failure)')
        plt.ylabel('Rotational Speed [rpm]')
        plt.grid(True)
        plt.savefig('../results/visualization/toolwear_failure.png')
        plt.show()





if __name__=="__main__":
    df=pd.read_csv('../Dataset/cleaned_failure_data.csv')
    print(df.head())
    analyse_visualize=Analyse_Visualize(df)
    analyse_visualize.analyse_operating_ranges()
    #analyse_visualize.plot_tool_wear()
    #failures_by_quality, failures_by_cause=analyse_visualize.determining_causes_of_failure()
    #analyse_visualize.plot_failure_nonfailure_distrubution(failures_by_quality, failures_by_cause)
    analyse_visualize.understad_of_failure()