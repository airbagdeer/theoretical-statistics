import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Models.model_class import Model
from typing import List, Optional


class RandomForestModel(Model):

    def __init__(self):

        self.__min_samples_split: int = 10
        self.__random_state: int = 1
        self.__n_estimators: int = 50
        super().__init__()

    def __initialize_model(self) -> None:

        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            min_samples_split=self.min_samples_split,
                                            random_state=self.random_state)

    def fit(self, df_test: pd.DataFrame, df_team_strength: pd.DataFrame, general_league_data: pd.DataFrame,
            features: List[str], min_samples_split: Optional[int] = 10, random_state: Optional[int] = 1,
            n_estimators: Optional[int] = 50) -> None:

        self.log(message="Start fitting the Model", log_type='info')
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_estimators = n_estimators

        super().initialize_data_frames(df_test=df_test,
                                       df_team_strength=df_team_strength,
                                       df_general_league_data=general_league_data)

        super().preprocess_df()
        self.__initialize_model()
        X = self.feature_scaling(features=features)
        self.model.fit(X=X, y=self.general_league_data['FTRN'])

    @property
    def min_samples_split(self) -> int:
        return self.__min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, min_samples_split: int) -> None:
        self.__min_samples_split = min_samples_split

    @property
    def random_state(self) -> int:
        return self.__random_state

    @random_state.setter
    def random_state(self, random_state: int) -> None:
        self.__random_state = random_state

    @property
    def n_estimators(self) -> int:
        return self.__n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimators: int) -> None:
        self.__n_estimators = n_estimators
