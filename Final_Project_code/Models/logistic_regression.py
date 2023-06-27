import pandas as pd
from typing import List
from Models.model_class import Model
from sklearn.linear_model import LogisticRegression


class LogisticRegress(Model):

    def __init__(self): super().__init__()

    def __initialize_model(self) -> None: self.model = LogisticRegression()

    def fit(self, df_test: pd.DataFrame, df_team_strength: pd.DataFrame, general_league_data: pd.DataFrame,
            features: List[str]) -> None:

        self.log(message="Start fitting the Model", log_type='info')
        super().initialize_data_frames(df_test=df_test,
                                       df_team_strength=df_team_strength,
                                       df_general_league_data=general_league_data)
        super().preprocess_df()
        self.__initialize_model()
        X = self.feature_scaling(features=features)
        self.model.fit(X=X, y=general_league_data['FTRN'])
