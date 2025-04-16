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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
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

# set numpy seed
SEED = 9
np.random.seed(SEED)

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



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
