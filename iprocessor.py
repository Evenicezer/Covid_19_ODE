import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter



def add_day_name_column(df):

    df['Date'] = pd.to_datetime(df['Date'])
    def get_day_name(date):
        return date.strftime('%A')

    df['date_name'] = df['Date'].apply(get_day_name)

    return df
#--------------------------------------
def add_date_name_column(df):
    df['days'] = pd.Series(range(0, len(df)), index=df.index)#(range(1, len(df)+1)
    return df
#--------------------------------------
def smooth_sundays_ssmt(df):
    """calculate mean for saturday sunday monday and tuesday"""
    # Find rows with 'Sunday' in the 'date_name' column
    sunday_rows = df[df['date_name'] == 'Sunday']
    #print(sunday_rows)
    print(df.columns)

    for index, sunday_row in sunday_rows.iloc[2:-2].iterrows():
        previous_rows = df.loc[index - 2: index - 1]  # Take two rows up
        next_rows = df.loc[index : index + 2]  # Take two rows down
        #print(previous_rows['n_recovered'], previous_rows['n_confirmed'],previous_rows['n_death'] , sep='/n')


        # Calculate the average values for columns 'n_confirmed', 'n_recovered', and 'n_death'
        avg_c = (previous_rows['n_confirmed'] + next_rows['n_confirmed']).mean()
        #print(f'avg_c_before: {avg_c}')
        avg_r = sum(previous_rows['n_recovered'] + next_rows['n_recovered']) / len(previous_rows['n_recovered'] + next_rows['n_recovered'])
        avg_d = sum(previous_rows['n_death'] + next_rows['n_death']) / len(previous_rows['n_death'] + next_rows['n_death'])

        # Replace the values in the current row
        df.at[index, 'n_confirmed'] = avg_c
        df.at[index, 'n_recovered'] = avg_r
        df.at[index, 'n_death'] = avg_d
        #print(f'avg_c: {avg_d}')
    return df
#-----------------------------------------


def smooth_sundays_ssm(df):
    for index, sunday_row in df[df['date_name'] == 'Sunday'].iloc[1:].iterrows():
        previous_row = df.loc[index - 1]#saturday
        next_row = df.loc[index + 1]#sunday

        # Calculate the average values for columns 'n_confirmed', 'n_recovered', and 'n_death'
        avg_c = math.ceil((previous_row['n_confirmed']  + next_row['n_confirmed']) / 2)
        avg_r = math.ceil((previous_row['n_recovered']  + next_row['n_recovered']) / 2)
        avg_d = math.ceil((previous_row['n_death']  + next_row['n_death']) / 2)

        # Replace the values in the current row
        df.at[index, 'n_confirmed'] = avg_c
        df.at[index, 'n_recovered'] = avg_r
        df.at[index, 'n_death'] = avg_d

    return df
#----------------------------------------


def smooth_sundays_rolling_ssm_w3(df):
    # Identify rows with Sunday in the 'date_name' column
    sunday_rows = df[df['date_name'] == 'Sunday']

    # Remove values in those rows
    #df.loc[sunday_rows.index, 'value'] = None

    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=3, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=3, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=3, center=True, min_periods=1, closed='right').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df.loc[sunday_rows.index, 'n_recovered'] = df.loc[sunday_rows.index, 'rolling_mean_r']
    df.loc[sunday_rows.index, 'n_confirmed'] = df.loc[sunday_rows.index, 'rolling_mean_c']
    df.loc[sunday_rows.index, 'n_death'] = df.loc[sunday_rows.index, 'rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    #df = df.drop(columns='rolling_mean')

    return df
#---------------------------------------
def smooth_sundays_rolling_ssm_w5(df):
    # Identify rows with Sunday in the 'date_name' column
    sunday_rows = df[df['date_name'] == 'Sunday']

    # Remove values in those rows
    #df.loc[sunday_rows.index, 'value'] = None

    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=5, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=5, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=5, center=True, min_periods=1, closed='right').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df.loc[sunday_rows.index, 'n_recovered'] = df.loc[sunday_rows.index, 'rolling_mean_r']
    df.loc[sunday_rows.index, 'n_confirmed'] = df.loc[sunday_rows.index, 'rolling_mean_c']
    df.loc[sunday_rows.index, 'n_death'] = df.loc[sunday_rows.index, 'rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    #df = df.drop(columns='rolling_mean')

    return df

#-----------------------------------------
def smooth_sundays_rolling_ssm_w3_smt(df):
    # Identify rows with Sunday in the 'date_name' column
    Tuesday_rows = df[df['date_name'] == 'Tuesday']
    sunday_rows = df[df['date_name'] == 'Sunday']

    # Remove values in those rows
    #df.loc[sunday_rows.index, 'value'] = None

    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=5, center=True, min_periods=1, closed='left').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=5, center=True, min_periods=1, closed='left').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=5, center=True, min_periods=1, closed='left').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df.loc[sunday_rows.index, 'n_recovered'] = df.loc[Tuesday_rows.index, 'rolling_mean_r']
    df.loc[sunday_rows.index, 'n_confirmed'] = df.loc[Tuesday_rows.index, 'rolling_mean_c']
    df.loc[sunday_rows.index, 'n_death'] = df.loc[Tuesday_rows.index, 'rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    #df = df.drop(columns='rolling_mean')
    return df
#-----------------------------------------
def smooth_sundays_rolling_w7_r(df):

    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=7, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=7, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=7, center=True, min_periods=1, closed='right').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df['n_recovered'] = df[ 'rolling_mean_r']
    df['n_confirmed'] = df[ 'rolling_mean_c']
    df['n_death'] = df['rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    #df = df.drop(columns='rolling_mean')

    return df

#---------------------------------------------
def smooth_sundays_rolling_w7_l(df):
    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=7, center=True, min_periods=1, closed='left').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=7, center=True, min_periods=1, closed='left').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=7, center=True, min_periods=1, closed='left').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df['n_recovered'] = df['rolling_mean_r']
    df['n_confirmed'] = df['rolling_mean_c']
    df['n_death'] = df['rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    # df = df.drop(columns='rolling_mean')

    return df


#----------------------------------------
def smooth_sundays_rolling_w5_r(df):

    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=5, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=5, center=True, min_periods=1, closed='right').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=5, center=True, min_periods=1, closed='right').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df['n_recovered'] = df[ 'rolling_mean_r']
    df['n_confirmed'] = df[ 'rolling_mean_c']
    df['n_death'] = df['rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    #df = df.drop(columns='rolling_mean')

    return df
#----------------------------------------#
def smooth_sundays_rolling_w5_l(df):

    # Apply the rolling method
    df['rolling_mean_r'] = df['n_recovered'].rolling(window=5, center=True, min_periods=1, closed='left').mean()
    df['rolling_mean_c'] = df['n_confirmed'].rolling(window=5, center=True, min_periods=1, closed='left').mean()
    df['rolling_mean_d'] = df['n_death'].rolling(window=5, center=True, min_periods=1, closed='left').mean()

    # Replace the values in Sunday rows with the calculated rolling mean
    df['n_recovered'] = df[ 'rolling_mean_r']
    df['n_confirmed'] = df[ 'rolling_mean_c']
    df['n_death'] = df['rolling_mean_d']

    # Drop the temporary 'rolling_mean' column
    #df = df.drop(columns='rolling_mean')

    return df
#-----------------------------------------------------
def plot_data_dcr(df, column_d: str, column_c: str, column_r: str, output_filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['days'], df[column_d], label='Daily Deaths', color='red')# use either Date or days
    ax.plot(df['days'], df[column_c], label='Daily Confirmed', color='blue')
    ax.plot(df['days'], df[column_r], label='Daily Recovered', color='orange')

    ax.set_xlabel('Days')
    ax.set_ylabel('Count')
    ax.set_title('COVID-19 Deaths and Cases Over Time')

    plt.xticks(rotation=45)
    #ax = plt.gca()
    #ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 30000, 2000)))

    ax.legend()

    plt.tight_layout()
    if output_filename:
        # Save the image to the specified output filename
        plt.savefig(output_filename)
    else:
        # Display the plot
        plt.show()
#---------------------------------------------------------------------------------

import matplotlib.pyplot as plt

def plot_data_dcr_multi(df,df3,df5,df7, column_:str,output_filename=None):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(df['days'], df[column_], label='origin', color='red')
    ax.plot(df['days'], df3[column_], label='w3_r', color='blue')
    ax.plot(df['days'], df5[column_], label='w5_r', color='orange')
    ax.plot(df['days'], df7[column_], label='w7_r', color='yellow')

    ax.set_xlabel('Days')
    ax.set_ylabel('Count')
    ax.set_title('COVID-19 Deaths and Cases Over Time')

    plt.xticks(rotation=45)

    ax.legend()

    plt.tight_layout()

    if output_filename:
        # Save the image to the specified output filename
        plt.savefig(output_filename)
    else:
        # Display the plot
        plt.show()




#----------------------------------------------------------------------------------------
if __name__=='__main__':
    plot_data_dcr()
    plot_data_dcr_multi()
    add_day_name_column()
    add_date_name_column()
    smooth_sundays_ssmt()
    smooth_sundays_ssm()
    smooth_sundays_rolling_ssm_w3()
    smooth_sundays_rolling_ssm_w5()
    smooth_sundays_rolling_ssm_w3_smt()
    smooth_sundays_rolling_w7_r()
    smooth_sundays_rolling_w5_r()
    smooth_sundays_rolling_w7_l()
    smooth_sundays_rolling_w5_l()