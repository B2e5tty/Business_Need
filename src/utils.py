import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats 
from scipy.stats import zscore

# finding missing values
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# def format_float(value):
#     return f'{value:,.2f}'


# def find_agg(df: pd.DataFrame, agg_column: str, agg_metric: str, col_name: str, top: int, order=False) -> pd.DataFrame:
#     new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name). \
#         sort_values(by=col_name, ascending=order)[:top]
#     return new_df


# function for converting bytes to megabytes
def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1 * 10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]


# function to remove outlier outside 95% of data distribution
def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
    return df[column]

# function to remove outliers
def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column], errors='ignore')

    return df

# funtion to plot distribution
def density_plot(df, column):
    df[column].plot(kind='density', subplots=True, layout=(2, 2), sharex=False, figsize=(15, 8))
    plt.suptitle('Density Plots of ' + column)
    plt.show()

# function to fill nan with mean    
def fill_mean(df, column):
    df[column] = df[column].fillna(df[column].mean())

# function to fill nan with median
def fill_median(df, column):
    df[column] = df[column].fillna(df[column].median())
    
# function to the outlier of columns
def box_plot(df, column):
    plt.figure(figsize=(13,6))
    sns.boxplot(data = df[column])
    plt.title(f"Box plot of :{column}")
    plt.show()

# function to convert miliseconds to mintues
def ms_to_mintue(x_ms):
    return x_ms * 0.0166667
 
# function for basic metrics calculation    
def metric_of_attributes(df):
    columns = df.columns[1:]
    max_value = []
    min_value =[]
    mean = []
    median = []
    users_above_mean = []
    users_below_mean = []
    users_below_median = []
    
    for i,col in enumerate(columns):
        max_value.append(df[col].max())
        min_value.append(df[col].min())
        
        mean.append(df[col].mean())                    # calculate mean
        median.append(df[col].median())                # calculate median
        
        users_above_mean.append(len(df[df[col] >= mean[i]]))
        users_below_mean.append(len(df[df[col] < mean[i]]))
                
    dic = {'Application': columns, 'Maximum':max_value,'Minimum':min_value,'Mean': mean,'Median': median, 'Above_Mean_Users': users_above_mean, 'Below_Mean_Users':users_below_mean}
    
    df = pd.DataFrame(dic)
    
    return df    

# function to calculate dispersion for attribute 
def dispersion_of_attributes(df):
    columns = df.columns[1:]
    range_value = []
    varience = []
    iqr = []
    std = []
    
    for i,col in enumerate(columns):
        range_value.append(df[col].max() - df[col].min())                    # calculate mean
        varience.append(df[col].var())                # calculate median
        iqr.append(df[col].quantile(0.75) - df[col].quantile(0.25))
        std.append(df[col].std()) 
                
    dic = {'Application': columns,'Range': range_value, 'Varience': varience,'IQR': iqr, 'Standard dev':std}
    df = pd.DataFrame(dic)
    
    return df    

# function to graph boxplot for univariate analysis
def graphical_univariate(df):
    columns = df.select_dtypes(include=np.number).columns[:-1]
    axisOne_columns = columns[:10]
    axisTwo_columns = columns[10:]
    fig, axs = plt.subplots(2,1,figsize = (20,8))
    axs = axs.flatten()

    axs[0].boxplot(df[axisOne_columns])
    axs[0].set_xticklabels(axisOne_columns)

    axs[1].boxplot(df[axisTwo_columns])
    axs[1].set_xticklabels(axisTwo_columns)
    plt.show()

    
# function to create dataframe that can be used for the bivariate analysis
def create_dataframe_for_bivariate(df):
    index = np.arange(0,18,1)
    remove = ['number_of_session','decile_class','Dur. (min)'] 
    columns = list(filter(lambda x: x not in remove, df.select_dtypes(include=np.number).columns))
    index_columns = list(zip(index,columns))
    
    columns_name = []
    new_df = []
    
    for i,col in index_columns:
        if i % 2 == 0 and i != 0:
            new_df.append(df[col] + df[index_columns[i-1][1]])
            columns_name.append((col.split('UL')[0]).strip())
        
    new_df = np.array(new_df)
    new_df = pd.DataFrame(new_df).T
    new_df.columns = columns_name
    
    new_df['Dur. (ms)'] = df['Dur. (ms)']
    new_df['Total Volume'] = df['Total Volume']
    new_df.rename(columns={'Total DL (Bytes)':'Total DL/UL'}, inplace = True)
    
    
    return new_df

# function for scatter plot 
def scatter_bivariate(df):
    remove = ['number_of_session','decile_class','Dur. (min)']     # columns not to be included on the bivariate analysis
    columns = list(filter(lambda x: x not in remove, df.select_dtypes(include=np.number).columns))     # filter the columns
    
    fig, axs = plt.subplots(5,2,figsize = (10,11))
    axs = axs.flatten()
    
    for i,col in enumerate(columns):
        axs[i].scatter(df[col],df['Total Volume'],alpha=0.2)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Total Volume')
    
    plt.tight_layout()
    plt.show()

# function to calculate the engagement metrics of users       
def engagement_stats(df):
    columns = df.select_dtypes(include = np.number).columns[:-1]
    
    max_value = []
    min_value = []
    total = []
    mean = []
    
    for col in columns:
        max_value.append(df[col].max())
        min_value.append(df[col].min())        
        total.append(df[col].sum())        
        mean.append(df[col].mean())             
        
    dic = {'Maximum':max_value, 'Minimum':min_value, 'Total':total,'Average':mean}
    
    df = pd.DataFrame(dic, index = ['Total Volume', 'number_of_session', 'Dur. (ms)'])
    
    return df    


# function for getting the top  10 users per applications
def engaged_users_per_application(df):
    applications = ['Social Media','Youtube','Netflix','Google','Email','Gaming','Other']

    for app in applications:
        new_df = df[['Customers',app]].sort_values(by = app, ascending = False)
        top_ten_customers = new_df['Customers'].head(10).values.tolist()

        print(f'The top ten customers of {app} application')
        print(f'{top_ten_customers} \n')
        
        
        
# functions for converting DL and UL columns into one
def create_dataframe_DLUL(df):
    index = np.arange(0,7,1)
    columns = df.select_dtypes(include=np.number).columns
    index_columns = list(zip(index,columns))
    columns_name = []
    new_df = []
    
    for i,col in index_columns:
        if i == 5:
            new_df.append((df[col] + df[index_columns[i+1][1]]) /2)
            columns_name.append((col.split('DL')[0]).strip())

        elif i % 2 != 0 and (i > 0 and i < 5):
            new_df.append((df[col] + df[index_columns[i+1][1]]) / 2)
            columns_name.append((col.split('DL')[0]).strip())
            
        # elif i >= 5 and i % 2 != 0:
        #     new_df.append(df[col] + df[index_columns[i+1][1]])
        #     columns_name.append((col.split('UL')[0]).strip())

    new_df = np.array(new_df)
    new_df = pd.DataFrame(new_df).T
    new_df.columns = columns_name
    
    new_df['MSISDN/Number'] = df['MSISDN/Number']
    new_df['Handset Type'] = df['Handset Type']
    new_df['Dur. (ms)'] = df['Dur. (ms)']
    new_df.rename(columns={'TCP':'TCP DL/UL'}, inplace = True)
    new_df.rename(columns={'Avg Bearer TP':'Avg Bearer TP DL/UL'}, inplace = True)
    new_df.rename(columns={'Avg RTT':'Avg RTT DL/UL'}, inplace = True)
    
    

    # new_df.rename(columns={'Total DL (Bytes)':'Total DL/UL'}, inplace = True)
    
    return new_df

# function to plot the dispersion of top 20 handset types
def dispersion_per_handsetType(df,col):
    top20_handset = df.groupby('Handset Type')[col].mean().sort_values(ascending = False).head(20)
    top20_handset_name = top20_handset.index
    
    print(f"The top 20 Handset type: {top20_handset}")
    handset_general_experience = df[df['Handset Type'].isin(top20_handset_name)]
    
    color_palette = [
        "#FF5733", "#33FF57", "#3357FF", "#F39C12", "#8E44AD",  # Bright Red, Green, Blue, Orange, Purple
        "#FF33FF", "#33FFFF", "#FF5733", "#FFC300", "#C70039",  # Magenta, Cyan, Red, Yellow, Maroon
        "#581845", "#900C3F", "#DAF7A6", "#900C3F", "#FFC300",  # Deep Purple, Burgundy, Light Green, Deep Maroon, Bright Yellow
        "#1F618D", "#117A65", "#D68910", "#2ECC71", "#2980B9"   # Navy Blue, Teal, Amber, Emerald Green, Royal Blue
    ]
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=handset_general_experience, x=col, hue='Handset Type', bins=30, palette=color_palette)
    plt.title('Distribution of Average Throughput for Top 20 Handset Types')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()    
