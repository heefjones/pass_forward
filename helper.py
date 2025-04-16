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
ROOT = './data/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_pass_data():
    # get all csv files in the 'pass_data' directory
    pass_paths = [os.path.join('./data/pass_data', file) for file in os.listdir('./data/pass_data') if file.endswith('.csv')]

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
        
        # get year as string from filename
        year = file_path[-8:-4]

        # add year column
        pass_data['year'] = year
        
        # add df to list
        pass_dfs.append(pass_data)

    # return the stacked dataframes
    return pd.concat(pass_dfs, axis=0)
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_rush_data():
    # get all csv files in the 'rush_data' directory
    rush_paths = [os.path.join('./data/rush_data', file) for file in os.listdir('./data/rush_data') if file.endswith('.csv')]

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
        year = file_path[-8:-4]
        
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

def show_unique_vals(df):
    """
    Print the number of unique values for each column in a DataFrame.
    If a column has fewer than 20 unique values, print those values.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # iterate over columns
    for col in df.columns:
        # get number of unique values and print
        n = df[col].nunique()
        print(f'"{col}" has {n} unique values')

        # if number of unique values is under 20, print the unique values
        if n < 20:
            print(df[col].unique())
        print()