import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import poisson
from Models.model_class import Model
from Utils.utils import get_info_from_df_strength_by_team_name


class PoissonModel(Model):
    """
    This Model predicts the results of a football match based on team_strength dataframe.
    """

    def __init__(self): super().__init__()

    def predict_match(self, home_team: str, away_team: str) -> int:
        """
        This function predicts the odd of the 2 teams.
        :param home_team: (str) name of the home team
        :param away_team: (str) name of the away team
        :return:
        """

        if all(team_name in self.df_team_strength.index for team_name in [home_team, away_team]):

            lamb_home_goals = self.df_team_strength.at[home_team, 'GoalsScored'] * \
                              self.df_team_strength.at[away_team, 'GoalsConceded']

            lamb_away_goals = self.df_team_strength.at[away_team, 'GoalsScored'] * \
                              self.df_team_strength.at[home_team, 'GoalsConceded']

            prob_goals_home, prob_goals_draw, prob_goals_away = self.__calculate_goals_probabilities(
                lamb_home=lamb_home_goals, lamb_away=lamb_away_goals, home_team=home_team, away_team=away_team)

            points_home = 3 * prob_goals_home + prob_goals_draw
            points_away = 3 * prob_goals_away + prob_goals_draw
            points_draw = 3 - (points_home + points_away)
            return int(np.argmax([points_draw, points_home, points_away]))

        else:
            return 0

    def __calculate_goals_probabilities(self, lamb_home: float, lamb_away: float,
                                        home_team: str, away_team: str) -> Tuple:
        """
        This function calculates the probabilities for victory/draw/loss in the match given 2 teams.
        :param lamb_home:
        :param lamb_away:
        :param home_team:
        :param away_team:
        :return:
        """

        home_team_shots, home_xg, home_xg_prob = get_info_from_df_strength_by_team_name(team_name=home_team,
                                                                                        df=self.df_team_strength)

        away_team_shots, away_xg, away_xg_prob = get_info_from_df_strength_by_team_name(team_name=away_team,
                                                                                        df=self.df_team_strength)

        prob_home, prob_away, prob_draw = 0, 0, 0
        for x in range(0, 11):  # number of goals home team
            for y in range(0, 11):  # number of goals away team

                p = poisson.pmf(x, lamb_home/home_team_shots*home_xg*home_xg_prob) * \
                    poisson.pmf(y, lamb_away/away_team_shots*away_xg*away_xg_prob)

                if x == y:
                    prob_draw += p
                elif x > y:
                    prob_home += p
                else:
                    prob_away += p

        return prob_home, prob_draw, prob_away

    def predict_all(self, df_test: pd.DataFrame, df_team_strength: pd.DataFrame,
                    general_league_data: pd.DataFrame) -> None:
        """
        This function goes over the entire bunch of matches in the dataframe and predicts the odds/results for each
        match.
        In addition the function compare the prediction vs real-results and calculates the accuracy.
        :return:
        """

        super().initialize_data_frames(df_test=df_test,
                                       df_team_strength=df_team_strength,
                                       df_general_league_data=general_league_data)

        self.df_test['FTR'].replace({'A': 2, 'H': 1, 'D': 0}, inplace=True)
        self.predictions = self.df_test[['HomeTeam', 'AwayTeam']].apply(lambda x: self.predict_match(*x), axis=1)
        self.df_test['FTR'].replace({'A': 2, 'H': 1, 'D': 0}, inplace=True)
        self.set_accuracy(labels=self.df_test['FTR'])
        self.set_f1_score(labels=self.df_test['FTR'])
        self.log(message=f"Accuracy = {self.accuracy} % ", log_type='info')
        self.log(message=f"F1_score = {self.f1_score} % ", log_type='info')
        self.log(message="Done")

