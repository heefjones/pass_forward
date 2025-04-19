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
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
COLORS = ['#00c9ff', '#005bff', '#006d8b', '#1f497d', '#000000']
TEAM_COLORS = {
    'GB': '#203731', 'CHI': '#0B162A', 'DET': '#0076B6', 'MIN': '#4F2683', 
    'NO': '#D3BC8D', 'ATL': '#A71930', 'CAR': '#0085CA', 'TB': '#D50A0A', 
    'LA': '#003594', 'SEA': '#002244', 'ARZ': '#97233F', 'SF': '#AA0000', 
    'DAL': '#041E42', 'NYG': '#0B2265', 'PHI': '#004C54', 'WAS': '#773141', 
    'LV': '#000000', 'LAC': '#0080C6', 'KC': '#E31837', 'DEN': '#FB4F14', 
    'BUF': '#00338D', 'NE': '#002244', 'MIA': '#008E97', 'NYJ': '#125740', 
    'CIN': '#FB4F14', 'PIT': '#FFB612', 'BLT': '#241773', 'CLV': '#311D00',
    'TEN': '#0C2340', 'JAX': '#006778', 'HST': '#002244', 'IND': '#003594'}

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_pass_data():
    """
    Load passing data from CSV files in the 'pass_data' directory.

    Returns:
    - (pd.DataFrame): A concatenated DataFrame containing passing statistics for quarterbacks.
    """

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
    """
    Load rushing data from CSV files in the 'rushing_data' directory.

    Returns:
    - (pd.DataFrame): A concatenated DataFrame containing rushing statistics for quarterbacks.
    """

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

def init_exp_col(df):
    """
    Initialize the experience column for each player in the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing player data.

    Returns:
    - df (pd.DataFrame): The DataFrame with the experience column initialized.
    """

    # experience of each player in 2006
    exp_2006 = {'Brett Favre': 15, 'Jon Kitna': 10, 'Marc Bulger': 6, 'Peyton Manning': 8,  'Drew Brees': 5, 'Eli Manning': 2, 'Carson Palmer': 3, 
                'Tom Brady': 6, 'Chad Pennington': 6, 'Rex Grossman': 3, 'Ben Roethlisberger': 2, 'Steve McNair': 11, 'Philip Rivers': 2, 'David Carr': 4, 
                'Alex Smith': 1, 'Brad Johnson': 14, 'Jake Delhomme': 9, 'J.P. Losman': 2, 'Charlie Frye': 1, 'Joey Harrington': 4, 'Michael Vick': 5, 
                'Matt Leinart': 0, 'Matt Hasselbeck': 8, 'Vince Young': 0, 'Tony Romo': 3, 'Bruce Gradkowski': 0, 'Jake Plummer': 9, 'Donovan McNabb': 7,
                'Andrew Walter': 1, 'Mark Brunell': 13, 'Damon Huard': 10, 'David Garrard': 4, 'Jason Campbell': 1, 'Trent Green': 13, 'Aaron Brooks': 7,
                'Jeff Garcia': 7, 'Byron Leftwich': 3, 'Kurt Warner': 12, 'Drew Bledsoe': 13, 'Seneca Wallace': 3, 'Jay Cutler': 0, 'Daunte Culpepper': 7, 
                'Derek Anderson': 1, 'Chris Simms': 3, 'Tim Rattay': 6, 'Chris Weinke': 5, 'Kerry Collins': 11, 'Tarvaris Jackson': 0, 'Cleo Lemon': 4, 
                'Kyle Boller': 3, 'Charlie Batch': 8, 'Sage Rosenfels': 5, 'A.J. Feeley': 5, 'Brian Griese': 8, 'Matt Schaub': 2, 'Jamie Martin': 13, 
                'Quinn Gray': 4, 'Brooks Bollinger': 3, 'Aaron Rodgers': 1, 'Marques Tuiasosopo': 5, 'Brett Basanez': 0, 'Matt Cassel': 1, 'Brodie Croyle': 0, 
                'Vinny Testaverde': 19, 'Gus Frerotte': 12, 'Anthony Wright': 7, 'Billy Volek': 6, 'Ken Dorsey': 3, 'Patrick Ramsey': 4, 'Kellen Clemens': 0}
    
    # add experience col to df
    df['exp'] = np.nan

    # add experience for 2006 QBs
    df.loc[df['year'] == 2006, 'exp'] = df.loc[df['year'] == 2006, 'player'].map(exp_2006)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def fill_experience(group):
    """
    Fill the experience column for each player in the group.

    Args:
    - group (pd.Series): The DataFrame containing player data for a specific player.

    Returns:
    - group (pd.Series): The DataFrame with the experience column filled.
    """

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
    - df (pd.DataFrame): DataFrame containing offensive grades.

    Returns:
    - None
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

# xgboost.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def xgb_cv(max_depth, learning_rate, gamma, min_child_weight, subsample, colsample_bytree, X, y):
    """
    Objective for Bayesian optimization of XGBRegressor. Returns the mean negative RMSE from CV.

    Args:
    - Hyperparameters to tune: max_depth, learning_rate, gamma, min_child_weight, subsample, colsample_bytree
    - X (pd.DataFrame): Features.
    - y (pd.Series): Lanels.

    Returns:
    - scores.mean (float) : mean of neg_root_mean_squared_error (larger is better).
    """

    # define XGBoost parameters
    params = {'n_estimators': 100,
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'min_child_weight': int(min_child_weight),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': SEED}

    # define model
    xgb = XGBRegressor(**params)

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(xgb, X, y, cv=kf, scoring='neg_root_mean_squared_error')

    # return mean cv score
    return scores.mean()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_2024_preds(preds_df):
    """
    Visualize the model's predictions against the true values.

    Args:
    - preds_df (pd.DataFrame): DataFrame containing the model's predictions and true values.

    Returns:
    - None
    """

    # visualize predictions
    plt.figure(figsize=(14, 8))

    # lists for annotating player names
    over_drops = ['Tim Boyle', 'Tommy DeVito', 'Dorian Thompson-Robinson', 'Kirk Cousins']
    under_drops = ['Nick Mullens', 'Mason Rudolph', 'Brock Purdy', 'Tyson Bagent', 
                'Davis Mills', 'Tyrod Taylor', 'Jimmy Garoppolo', 'Daniel Jones', 'Russell Wilson', 'Kenny Pickett', 'Joe Flacco']

    # title, labels
    plt.title('2024 Predictions with XGBoost', fontsize=22)
    plt.xlabel('True Offensive Grade', fontsize=22)
    plt.ylabel('Predicted Offensive Grade', fontsize=22)

    # team colors
    color_palette = {color: color for color in preds_df['color'].unique()}

    # plot players as points
    sns.scatterplot(data=preds_df, x='y_true', y='y_pred', hue='color', palette=color_palette, legend=False)

    # plot line to show perfect predictions
    sns.lineplot(x=range(30,96), y=range(30,96), color='black')

    # annotating each point with the player's name
    for index, row in preds_df.iterrows():
        # over preds (above line)
        if (row['y_pred'] > row['y_true']) and (row['player'] not in over_drops):
            plt.text(row['y_true']-0.5, row['y_pred']-0.3, row['player'], horizontalalignment='right', color='black', 
                weight='semibold', fontsize=7)
        # under preds (below line)
        elif (row['y_pred'] < row['y_true']) and (row['player'] not in under_drops):
            plt.text(row['y_true']+0.5, row['y_pred']-0.3, row['player'], horizontalalignment='left', color='black', 
                weight='semibold', fontsize=7)
            
    # annotate "Over-predictions" and "Under-predictions"
    plt.text(35, 85, 'Over-predictions', fontsize=20, weight='semibold', color='red')
    plt.text(75, 45, 'Under-predictions', fontsize=20, weight='semibold', color='red')
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_2025_preds(preds_df):
    """
    Visualize the model's 2025 predictions.

    Args:
    - preds_df (pd.DataFrame): DataFrame containing the model's predictions.

    Returns:
    - None
    """

    # reverse the df for proper display
    preds_df = preds_df[::-1]

    # plot
    plt.figure(figsize=(14, 10))
    plt.hlines(y=preds_df['player_with_rank'], xmin=0, xmax=preds_df['y_pred'], color=preds_df['color'], lw=5)
    plt.plot(preds_df['y_pred'], preds_df['player_with_rank'], 'o', color='black')
    plt.title('2025 Predictions with XGBoost (Non-Rookies)', fontsize=22)
    plt.xlabel('Predicted Offensive Grade', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlim(60, 85)
    plt.margins(x=0, y=0.03)

    # annotate
    for i, row in preds_df.iterrows():
        plt.text(row['y_pred'] + 0.3, row['player_with_rank'], f'{row["y_pred"]:.1f}', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# rnn.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_sequences(df):
    """
    Create sequences of features and labels for each player.

    Args:
    - df (pd.DataFrame): contains player features and labels.

    Returns:
    - X_pad (torch.Tensor): shape (n_players, max_seasons, n_features).
    - y (torch.Tensor): shape (n_players,).
    - lengths (torch.Tensor): shape (n_players,).
    - mask (torch.Tensor): shape (n_players, max_seasons).
    - players (list): list of player names.
    """

    # non-feature columns
    non_feat_cols = ['player', 'team_name', 'year', 'target']

    # init lists
    sequences, labels, players = [], [], []

    # iterate through each player
    for player, g in df.groupby('player'):
        # sort
        g = g.sort_values('year').reset_index(drop=True)

        # cache the feature matrix once
        feat_mat = g.drop(columns=non_feat_cols).values

        # iterate through each season
        for i in range(len(g)):
            # seasons 0 through i
            seq = torch.tensor(feat_mat[:i+1], dtype=torch.float32)
            
            # target for season i
            lbl = torch.tensor(g['target'].iloc[i], dtype=torch.float32)

            # append to lists
            sequences.append(seq)
            labels.append(lbl)
            players.append(player)

    # pad to longest sequence
    X_pad = pad_sequence(sequences, batch_first=True)  

    # build mask so model knows which timesteps are real
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    max_len = X_pad.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    # create labels
    y = torch.stack(labels)

    return X_pad, y, lengths, mask, players

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class SeqDataset(Dataset):
    """
    Dataset class for sequences of features and labels.
    """
    
    # initialize the dataset
    def __init__(self, X, lengths, y):
        self.X, self.lengths, self.y = X, lengths, y

    # get the length of the dataset
    def __len__(self): return len(self.y)

    # get an item from the dataset
    def __getitem__(self, i): return self.X[i], self.lengths[i], self.y[i]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class RNNRegressor(nn.Module):
    """
    A simple RNN regressor for predicting player performance based on sequences of features.
    """
    
    # initialize the model
    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.rnn = nn.RNN(in_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    # forward pass
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hn = self.rnn(packed)
        out = hn[-1]  # final hidden state for the last layer
        return self.head(out).squeeze(1)
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def train_rnn(model, opt, criterion, train_dl, val_dl, train_ds, device):
    """
    Train the RNN model with early stopping.

    Args:
    - model (nn.Module): The RNN model to train.
    - opt (torch.optim.Optimizer): The optimizer for training.
    - criterion (nn.Module): The loss function.
    - train_dl (DataLoader): DataLoader for the training set.
    - val_dl (DataLoader): DataLoader for the validation set.
    - train_ds (Dataset): The training dataset.
    - device (torch.device): The device to train on (CPU or GPU).

    Returns:
    - None
    """

    # init vars for early stopping
    best_rmse = float('inf')
    patience = 100
    wait = 0
    best_model_state = None

    # training loop
    for epoch in range(1000):
        # training
        model.train()
        train_loss = 0
        for Xb, lb, yb in train_dl:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            preds = model(Xb, lb)
            loss  = criterion(preds, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()*Xb.size(0)
        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for Xb, lb, yb in val_dl:
                Xb = Xb.to(device)
                yb = yb.to(device)
                vp = model(Xb, lb)
                val_preds.append(vp.cpu())
                val_trues.append(yb.cpu())
        val_preds = torch.cat(val_preds).numpy()
        val_trues = torch.cat(val_trues).numpy()

        # evaluate
        rmse = mean_squared_error(val_trues, val_preds, squared=False)
        r2 = r2_score(val_trues, val_preds)

        # print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train MSE: {train_loss:.3f} | Val RMSE: {rmse:.3f}, R²: {r2:.3f}")

        # early stopping logic
        if rmse < best_rmse:
            best_rmse = rmse
            wait = 0
            best_model_state = model.state_dict()  # save best model
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.3f}")
                break

    # load best model back
    if best_model_state is not None:
        model.load_state_dict(best_model_state)