import pickle5 as pickle
import pandas as pd
import numpy as np
from typing import Tuple, Any
import scipy.stats as stats


def sigmoid(x: Any) -> Any:

    return 1/(1+np.exp(-x))


def load_data(data_name: str) -> pd.DataFrame:
    """
    This function loads a dataframe.
    :param data_name: (str) name of the dataset.
    :return: (pd.DataFrame) a data frame.
    """
    with open(data_name, 'rb') as f:
        return pickle.load(f)


def get_info_from_df_strength_by_team_name(team_name: str, df: pd.DataFrame) -> Tuple:
    """
    This function gets info from df_strength by team's name
    :param team_name:
    :param df:
    :return:
    """
    return df.loc[team_name]['Shots'], df.loc[team_name]['xG'], df.loc[team_name]['xG_prob']


def run_hypothesis(df_team_strength: pd.DataFrame, data: pd.DataFrame) -> Tuple:
    """
    This function apply the t-Test on the hypothesis that says whether or not a high team strength
    increases the likelihood to win a football match.
    :param df_team_strength:
    :param data:
    :return:
    """

    total_matches_played: int = 152
    diff_strength_rate = df_team_strength['GoalsScored'] - df_team_strength['GoalsConceded'] + \
                         df_team_strength['Shots'] * df_team_strength['xG_prob']

    teams = data['HomeTeam'].unique().tolist()
    teams.sort()

    total_win_ratio = []

    for team in teams:
        home_ratio = data[(data['HomeTeam'] == team) & (data['FTR'] == 'H')].shape[0]
        away_ratio = data[(data['AwayTeam'] == team) & (data['FTR'] == 'A')].shape[0]

        total_win_ratio.append((home_ratio + away_ratio) / total_matches_played * 100)

    return stats.ttest_ind(diff_strength_rate, total_win_ratio, equal_var=False)


def create_df_strength(data: pd.DataFrame) -> pd.DataFrame:

    if "HomeGoals" not in data.columns or "AwayGoals" not in data.columns:

        data.rename(columns={'score1': 'HomeGoals', 'score2': 'AwayGoals'}, inplace=True)

    df_home = data[['HomeTeam', 'HomeGoals', 'AwayGoals', 'HST', 'xg1']].copy()
    df_home.rename(columns={'HomeTeam': 'Team', 'HomeGoals': 'GoalsScored', 'AwayGoals': 'GoalsConceded',
                            'HST': 'Shots', 'xg1': 'xG'},
                   inplace=True)

    df_away = data[['AwayTeam', 'HomeGoals', 'AwayGoals', 'AST', 'xg2']].copy()
    df_away.rename(columns={'HomeTeam': 'Team', 'HomeGoals': 'GoalsConceded', 'AwayGoals': 'GoalsScored',
                            'AST': 'Shots', 'xg2': 'xG'},
                   inplace=True)

    df_team_strength = pd.concat([df_home, df_away], ignore_index=True).groupby('Team').mean()
    df_team_strength['xG_prob'] = sigmoid(df_team_strength['xG'].values)
    return df_team_strength
