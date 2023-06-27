import pandas as pd
import numpy as np
from typing import Any, Tuple, List
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from loguru import logger


class Model:

    def __init__(self):
        self.__df_team_strength = pd.DataFrame()  # contains the scoring rate strength for each team
        self.__general_league_data = pd.DataFrame()  # contains the entire data for a certain league
        self.__df_test = pd.DataFrame()
        self.__model: list = []
        self.__predictions: list = []
        self.__accuracy: float = 0
        self.__f1_score: float = 0
        self.__roc_auc_score: float = 0
        self.__X_test = pd.DataFrame()

    def initialize_data_frames(self, df_test: pd.DataFrame,
                               df_team_strength: pd.DataFrame,
                               df_general_league_data: pd.DataFrame):
        self.__df_team_strength = df_team_strength
        self.__df_test = df_test
        self.__general_league_data = df_general_league_data

    def log(self, message: str, log_type: str = 'success') -> None:

        if log_type == 'info':
            logger.info(message)
        else:
            logger.success("Done")

    def set_accuracy(self, labels: Any) -> None:
        if len(self.predictions) == len(labels):
            self.accuracy = accuracy_score(labels, self.predictions) * 100
        else:
            raise Exception("Dimension Error<!> accuracy was not calculated <!>")

    def set_f1_score(self, labels: Any) -> None:

        if len(self.predictions) == len(labels):
            self.f1_score = f1_score(labels, self.predictions, average='weighted')
        else:
            raise Exception("Dimension Error<!> f1_score was not calculated <!>")

    def __insert_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        home_str = []
        away_str = []
        for row in range(df.shape[0]):
            home_team = df.iloc[row]['HomeTeam']
            away_team = df.iloc[row]['AwayTeam']

            if all(team_name in self.df_team_strength.index for team_name in [home_team, away_team]):

                home_team_info = self.df_team_strength.loc[home_team]
                away_team_info = self.df_team_strength.loc[away_team]

                home_str.append(home_team_info['xG_prob'] * home_team_info['Shots'] +
                                home_team_info['GoalsScored'] - home_team_info['GoalsConceded'])

                away_str.append(away_team_info['xG_prob'] * away_team_info['Shots'] +
                                away_team_info['GoalsScored'] - away_team_info['GoalsConceded'])

            else:
                home_str.append(0)
                away_str.append(0)

        df['HomeStrength'] = home_str
        df['AwayStrength'] = away_str

        return df

    def preprocess_df(self):

        self.general_league_data['FTRN'] = self.general_league_data['FTR'].replace({'A': 2, 'D': 0, 'H': 1},
                                                                                   inplace=True)
        self.general_league_data['HomeTeamCat'] = self.general_league_data['HomeTeam'].astype("category").cat.codes
        self.general_league_data['AwayTeamCat'] = self.general_league_data['AwayTeam'].astype("category").cat.codes

        self.df_test['FTRN'] = self.df_test['FTR'].replace({'A': 2, 'D': 0, 'H': 1}, inplace=True)
        self.df_test['HomeTeamCat'] = self.df_test['HomeTeam'].astype("category").cat.codes
        self.df_test['AwayTeamCat'] = self.df_test['AwayTeam'].astype("category").cat.codes

        self.general_league_data = self.__insert_strength(df=self.general_league_data)
        self.df_test = self.__insert_strength(df=self.df_test)

    def predict(self):

        self.predictions = self.model.predict(X=self.X_test)
        self.set_accuracy(labels=self.df_test['FTRN'])
        self.set_f1_score(labels=self.df_test['FTRN'])
        logger.info(f"Accuracy = {self.accuracy} %")
        logger.info(f"F1_score = {self.f1_score} %")
        logger.success("Done")

    def feature_scaling(self, features: List) -> Tuple:

        scaler = StandardScaler()
        X = scaler.fit_transform(self.general_league_data[features])
        self.X_test = scaler.transform(self.df_test[features])
        return X

    @property
    def df_team_strength(self) -> pd.DataFrame:
        return self.__df_team_strength

    @df_team_strength.setter
    def df_team_strength(self, df_team_strength: pd.DataFrame) -> None:
        self.__df_team_strength = df_team_strength

    @property
    def general_league_data(self) -> pd.DataFrame:
        return self.__general_league_data

    @general_league_data.setter
    def general_league_data(self, general_league_table: pd.DataFrame) -> None:
        self.__general_league_data = general_league_table

    @property
    def df_test(self) -> pd.DataFrame:
        return self.__df_test

    @df_test.setter
    def df_test(self, df_test: pd.DataFrame) -> None:
        self.__df_test = df_test

    @property
    def model(self) -> Any:
        return self.__model

    @model.setter
    def model(self, model: Any) -> None:
        self.__model = model

    @property
    def predictions(self) -> Any:
        return self.__predictions

    @predictions.setter
    def predictions(self, predictions: Any) -> None:
        self.__predictions = predictions

    @property
    def accuracy(self) -> float:
        return self.__accuracy

    @accuracy.setter
    def accuracy(self, accuracy: float) -> None:
        self.__accuracy = accuracy

    @property
    def X_test(self) -> pd.DataFrame:
        return self.__X_test

    @X_test.setter
    def X_test(self, X_test: pd.DataFrame) -> None:
        self.__X_test = X_test

    @property
    def f1_score(self) -> float:
        return self.__f1_score

    @f1_score.setter
    def f1_score(self, f1_score: float) -> None:
        self.__f1_score = f1_score

    @property
    def roc_auc_score(self) -> float:
        return self.__roc_auc_score

    @roc_auc_score.setter
    def roc_auc_score(self, roc_auc_score: float) -> None:
        self.__roc_auc_score = roc_auc_score
