import pandas as pd
import numpy as np
import os
from itertools import permutations
from Utils.utils import load_data
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import gamma


class MonteCarlo:

    def __init__(self, data_to_simulate: pd.DataFrame, num_of_sim_per_match: int):

        self.__data_to_simulate = data_to_simulate
        self.__num_of_sim = num_of_sim_per_match

    def fit_distribution(self, column: str, team: str, is_home: bool):

        data = self.__data_to_simulate[(self.__data_to_simulate['HomeTeam'] == team)] if is_home \
            else self.__data_to_simulate[(self.__data_to_simulate['AwayTeam'] == team)]

        f = Fitter(data[f'{column}'].values,
                   distributions=['gamma'])
        f.fit()
        return f.fitted_param['gamma'][0]

    def __simulate(self, homeTeam: str, awayTeam: str) -> pd.DataFrame:

        home_shots_params = self.fit_distribution(column='HST', team=homeTeam, is_home=True)
        away_shots_params = self.fit_distribution(column='AST', team=awayTeam, is_home=False)
        home_xg_params = self.fit_distribution(column='xg1', team=homeTeam, is_home=True)
        away_xg_params = self.fit_distribution(column='xg2', team=awayTeam, is_home=False)

        res = pd.DataFrame()
        for sim_num in range(self.__num_of_sim):
            home_goals: int = 0
            away_goals: int = 0

            print(f"HomeTeam = {homeTeam} vs AwayTeam = {awayTeam} , {sim_num}")
            if not self.__data_to_simulate.empty:

                home_shots = np.abs(int(np.random.gamma(home_shots_params)))
                away_shots = np.abs(int(np.random.gamma(away_shots_params)))

                gamma_dist_home = gamma(a=home_xg_params)
                home_xg = gamma_dist_home.rvs()
                home_xg_prob = gamma_dist_home.pdf(home_xg)

                gamma_dist_away = gamma(a=away_xg_params)
                away_xg = gamma_dist_away.rvs()
                away_xg_prob = gamma_dist_away.pdf(away_xg)

                for home_shot in range(home_shots):

                    random_shot_prob = gamma_dist_home.pdf(gamma_dist_home.rvs())
                    if random_shot_prob < home_xg_prob:
                        home_goals += 1

                for away_shot in range(away_shots):

                    random_shot_prob = gamma_dist_away.pdf(gamma_dist_away.rvs())

                    if random_shot_prob < away_xg_prob:
                        away_goals += 1

                if home_goals < 8 and away_goals < 8:
                    if home_goals > away_goals:
                        result = 1
                    elif home_goals == away_goals:
                        result = 0
                    else:
                        result = 2

                    generated_data = [
                        {'HomeTeam': homeTeam, 'AwayTeam': awayTeam, 'HST': home_shots, 'AST': away_shots,
                         'HomeGoals': home_goals, 'AwayGoals': away_goals, 'xG_home': home_xg, 'xG_Away': away_xg,
                         'FTR': result}]
                    res = pd.concat([res, pd.DataFrame(generated_data)])

        return res

    def generate_data(self) -> pd.DataFrame:

        generated_data = pd.DataFrame()
        teams = self.__data_to_simulate['HomeTeam'].unique()
        pers = permutations(teams, 2)
        for p in pers:
            generated_data = pd.concat([generated_data, self.__simulate(homeTeam=p[0], awayTeam=p[1])])

        return generated_data


if __name__ == "__main__":
    general_league_data = load_data('Premier_League_df_train')
    mc = MonteCarlo(data_to_simulate=general_league_data, num_of_sim_per_match=50)
    b = mc.generate_data()
    pd.to_pickle(b, f"{os.getcwd()}\\synthetic_data_{b.shape[0]}_samples")
