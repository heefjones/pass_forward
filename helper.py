# data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest

# system
import os
import sys
import warnings

# machine learning
from tqdm import tqdm
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
EXP_2006 = {'Brett Favre': 15, 'Jon Kitna': 10, 'Marc Bulger': 6, 'Peyton Manning': 8,  'Drew Brees': 5, 'Eli Manning': 2, 'Carson Palmer': 3, 
                'Tom Brady': 6, 'Chad Pennington': 6, 'Rex Grossman': 3, 'Ben Roethlisberger': 2, 'Steve McNair': 11, 'Philip Rivers': 2, 'David Carr': 4, 
                'Alex Smith': 1, 'Brad Johnson': 14, 'Jake Delhomme': 9, 'J.P. Losman': 2, 'Charlie Frye': 1, 'Joey Harrington': 4, 'Michael Vick': 5, 
                'Matt Leinart': 0, 'Matt Hasselbeck': 8, 'Vince Young': 0, 'Tony Romo': 3, 'Bruce Gradkowski': 0, 'Jake Plummer': 9, 'Donovan McNabb': 7,
                'Andrew Walter': 1, 'Mark Brunell': 13, 'Damon Huard': 10, 'David Garrard': 4, 'Jason Campbell': 1, 'Trent Green': 13, 'Aaron Brooks': 7,
                'Jeff Garcia': 7, 'Byron Leftwich': 3, 'Kurt Warner': 12, 'Drew Bledsoe': 13, 'Seneca Wallace': 3, 'Jay Cutler': 0, 'Daunte Culpepper': 7, 
                'Derek Anderson': 1, 'Chris Simms': 3, 'Tim Rattay': 6, 'Chris Weinke': 5, 'Kerry Collins': 11, 'Tarvaris Jackson': 0, 'Cleo Lemon': 4, 
                'Kyle Boller': 3, 'Charlie Batch': 8, 'Sage Rosenfels': 5, 'A.J. Feeley': 5, 'Brian Griese': 8, 'Matt Schaub': 2, 'Jamie Martin': 13, 
                'Quinn Gray': 4, 'Brooks Bollinger': 3, 'Aaron Rodgers': 1, 'Marques Tuiasosopo': 5, 'Brett Basanez': 0, 'Matt Cassel': 1, 'Brodie Croyle': 0, 
                'Vinny Testaverde': 19, 'Gus Frerotte': 12, 'Anthony Wright': 7, 'Billy Volek': 6, 'Ken Dorsey': 3, 'Patrick Ramsey': 4, 'Kellen Clemens': 0}
COLORS = ['#00c9ff', '#005bff', '#006d8b', '#1f497d', '#000000']
TEAM_COLORS = {
    'GB': '#203731',   # Green Bay Packers
    'NO': '#D3BC8D',   # New Orleans Saints
    'LA': '#003594',   # Los Angeles Rams
    'DAL': '#041E42',  # Dallas Cowboys
    'NYG': '#0B2265',  # New York Giants
    'LV': '#000000',   # Las Vegas Raiders
    'SEA': '#002244',  # Seattle Seahawks
    'PHI': '#004C54',  # Philadelphia Eagles
    'DET': '#0076B6',  # Detroit Lions
    'SF': '#AA0000',   # San Francisco 49ers
    'CIN': '#FB4F14',  # Cincinnati Bengals
    'BUF': '#00338D',  # Buffalo Bills
    'CHI': '#0B162A',  # Chicago Bears
    'LAC': '#0080C6',  # Los Angeles Chargers
    'PIT': '#FFB612',  # Pittsburgh Steelers
    'MIN': '#4F2683',  # Minnesota Vikings
    'ARZ': '#97233F',  # Arizona Cardinals
    'BLT': '#241773',  # Baltimore Ravens
    'NE': '#002244',   # New England Patriots
    'KC': '#E31837',   # Kansas City Chiefs
    'DEN': '#FB4F14',  # Denver Broncos
    'TEN': '#0C2340',  # Tennessee Titans
    'WAS': '#773141',  # Washington Commanders
    'JAX': '#006778',  # Jacksonville Jaguars
    'MIA': '#008E97',  # Miami Dolphins
    'NYJ': '#125740',  # New York Jets
    'TB': '#D50A0A',   # Tampa Bay Buccaneers
    'CLV': '#311D00',  # Cleveland Browns
    'ATL': '#A71930',  # Atlanta Falcons
    'CAR': '#0085CA',  # Carolina Panthers
}

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_pass_data():
    # get all csv files in the 'pass_data' directory
    pass_paths = [os.path.join('./data/passing', file) for file in os.listdir('./data/passing') if file.endswith('.csv')]

    # list to hold all dfs
    pass_dfs = []

    # read each file into a dataframe
    for file_path in pass_paths:
        # drop unnecessary/redundant stats
        drops = ['player_id', 'position', 'declined_penalties', 'franchise_id', 'grades_run']
        
        # load each season into a df, get QBs, drop specified columns
        pass_data = pd.read_csv(file_path).query("position == 'QB'").drop(drops, axis=1)
        
        # add 'pass' to cols to indicate passing stat
        pass_data.columns = ['player', 'team_name', 'player_game_count'] + ['pass_' + col for col in pass_data.columns[3:]]
        
        # get year from filename
        year = int(file_path[-8:-4])

        # add year column
        pass_data['year'] = year
        
        # add df to list
        pass_dfs.append(pass_data)

    # return the stacked dataframes
    return pd.concat(pass_dfs, axis=0)
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_rush_data():
    # get all csv files in the 'rush_data' directory
    rush_paths = [os.path.join('./data/rushing', file) for file in os.listdir('./data/rushing') if file.endswith('.csv')]

    # list to hold all dfs
    rush_dfs = []

    # read each file into a dataframe
    for file_path in rush_paths:
        # drop unnecessary/redundant stats
        drops = ['player_id', 'position', 'team_name', 'player_game_count', 'declined_penalties', 'drops', 'franchise_id', 
              'grades_hands_fumble', 'grades_offense', 'grades_pass','grades_pass_block', 'grades_pass_route', 
              'grades_run_block', 'penalties', 'rec_yards', 'receptions', 'routes', 'scrambles', 'targets', 'yprr']
        
        # load each season into a df, get QBs, drop specified columns
        rush_data = pd.read_csv(file_path).query("position == 'QB'").drop(drops, axis=1)
        
        # add 'rush' to cols to indicate rushing stat
        rush_data.columns = ['player'] + ['rush_' + col for col in rush_data.columns[1:]]

        # get year as string from filename
        year = int(file_path[-8:-4])
        
        # add year column
        rush_data['year'] = year
        
        # add df to list
        rush_dfs.append(rush_data)

    # return the stacked dataframes
    return pd.concat(rush_dfs, axis=0)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')

    # display null values
    display(df.isnull().sum().to_frame().T)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def fill_experience(group):
    # get first experience value for a player
    first_exp = group['exp'].iloc[0]
    
    # if value is null, set to 0 (rookie season)
    if pd.isna(first_exp):
        first_exp = 0
    
    # define range of years to fill each player's experience column
    experience = range(int(first_exp), int(first_exp) + len(group))
    group['exp'] = list(experience)
    return group

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_hist_with_annot(df, col, bins=None, xticklabels=None, vertical_lines=None, color='black'):
    """
    Plots a histogram of a column and optionally adds vertical lines with percentage annotations.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - col (str): Column name to plot.
    - bins (int, optional): Number of bins in the histogram. Default is the square root of the number of rows in the DataFrame.
    - xticklabels (list[int], optional): Custom x-tick labels. Default is None.
    - vertical_lines (list[int], optional): List of x-values where vertical lines should be drawn. Defaults to None.

    Returns:
    - None
    """

    # default bins (square root of number of rows)
    if bins is None:
        bins = int(np.sqrt(df.shape[0]))

    # get data
    data = df[col]

    # compute histogram
    counts, bin_edges = np.histogram(data, bins=np.arange(data.min(), data.max() + 2) - 0.5)

    # compute bin centers (which are your actual data values)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # plot with bars centered
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.bar(bin_centers, counts, width=1.0, align='center', color=color, edgecolor='black')

    # center X ticks under the bars
    ax.set_xticks(np.arange(data.min(), data.max() + 1))
    ax.set_xticklabels(np.arange(data.min(), data.max() + 1))

    # set x-tick labels
    if xticklabels:
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels)

    # add vertical lines and region annotations
    if vertical_lines:
        vertical_lines = sorted(vertical_lines)
        for x in vertical_lines:
            ax.axvline(x=x, color='red', linestyle='dashed', linewidth=2)

        total_count = len(data)
        prev_x = data.min()
        for x in vertical_lines + [data.max()]:
            region_pct = ((data >= prev_x) & (data < x)).sum() / total_count * 100
            ax.text((prev_x + x) / 2, max(counts) * 0.9, f'{region_pct:.1f}%',
                    color='black', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
            prev_x = x

    plt.show()
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_pairplot(df):
    """
    Plot pairplot of offensive grades.

    Args:
        df (pd.DataFrame): DataFrame containing offensive grades.
    """

    # rename columns for the plot
    df_rename = df.rename(columns={'pass_grades_offense': 'Offense', 'pass_grades_pass': 'Passing', 'rush_grades_run': 'Rushing'})

    # create pairplot
    g = sns.PairGrid(df_rename[['Offense', 'Passing', 'Rushing']])

    # map residual plot to upper triangle
    g.map_upper(sns.residplot, color=COLORS[2]) 

    # map regression plot to the lower triangle
    g.map_lower(sns.scatterplot, color=COLORS[3])

    # map histogram to the diagonal
    g.map_diag(sns.histplot, color=COLORS[4])

    # remove tick labels
    for ax in g.axes.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([]);

    # set title
    g.fig.suptitle('Pairplot of Offensive Grades', fontsize=16, fontweight='bold')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def agg_year_groups(df):
    """
    Create rolling mean columns for year groups and career mean.

    Args:
    - df (pd.DataFrame): DataFrame containing player data with 'pass_grades_offense' column.

    Returns:
    - df_agg (pd.DataFrame): DataFrame with additional columns for rolling means and career mean.
    """

    # create year groups
    year_groups = [1, 2, 3, 4, 5, 10]

    # initialize df_agg
    df_agg = df[['player', 'year', 'target', 'pass_grades_offense']].copy()

    # iterate through year groups
    for n in year_groups:
        col_name = f'pass_grade_offense_{n}year_mean'
        
        # create an empty column
        df_agg[col_name] = np.nan

        # loop over each player
        for player, group in df.groupby('player'):
            # get rolling mean for this player
            rolling_vals = (
                group['pass_grades_offense']
                .rolling(window=n, min_periods=n)
                .mean()
                .values)

            # set values back into the right rows
            df_agg.loc[group.index, col_name] = rolling_vals

    # add career mean up to each season (cumulative mean, shifted to exclude current season)
    df_agg['career_pass_grade_offense_mean'] = (df.groupby('player')['pass_grades_offense'].apply(lambda x: x.expanding().mean().shift()))

    return df_agg

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def compute_rolling_stats(df, agg_cols):
    """
    Compute rolling and career statistics for specified columns in the dataframe.

    Args:
    - df (pd.DataFrame): The input dataframe containing player statistics.
    - agg_cols (list): List of column names to compute rolling and career statistics for.

    Returns:
    - pd.DataFrame: The dataframe with new columns for rolling and career statistics.
    """

    # ensure values are sorted properly
    df = df.sort_values(by=['player', 'year']).reset_index(drop=True)

    # create empty list to collect new cols
    new_columns = []

    # loop through each agg col
    for col in agg_cols:
        # 5-year rolling mean and std
        rolling_mean = df.groupby('player')[col].rolling(window=5, min_periods=1).mean().astype('float16').reset_index(level=0, drop=True)
        rolling_std = df.groupby('player')[col].rolling(window=5, min_periods=1).std().astype('float16').reset_index(level=0, drop=True)

        # career mean and std
        career_mean = df.groupby('player')[col].expanding().mean().astype('float16').reset_index(level=0, drop=True)
        career_std = df.groupby('player')[col].expanding().std().astype('float16').reset_index(level=0, drop=True)

        # add new cols to list
        new_columns.extend([rolling_mean.rename(f'{col}_5y_mean'), rolling_std.rename(f'{col}_5y_std'), career_mean.rename(f'{col}_career_mean'), career_std.rename(f'{col}_career_std')])

    # concat the original df with the new columns
    df = pd.concat([df] + new_columns, axis=1)

    # fill nulls (not including target col)
    df.loc[:, df.columns != 'target'] = df.loc[:, df.columns != 'target'].fillna(0)

    # sort cols
    return df[sorted(df.columns)]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# preds.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
