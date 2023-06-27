import pandas as pd
from Models.model_factory import ModelFactory
from Utils.utils import load_data, run_hypothesis, create_df_strength


if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~ Loading the Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    data = load_data('Premier_League_df_train')
    df_test = load_data('Premier_League_df_test')

    general_league_data = load_data('synthetic_data_3139_samples')
    general_league_data = pd.concat([general_league_data, load_data('Premier_League_df_train')])
    df_team_strength = create_df_strength(data=general_league_data)
    features = ['HomeTeamCat', 'AwayTeamCat', 'HomeStrength', 'AwayStrength']
    models_names = ['RandomForest', 'Poisson', 'SVM', 'XGBoost', 'LogisticRegression']
    accuracy = []
    f1 = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Applying ML models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    models = ModelFactory()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~ RandomForest Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    rf = models.get_model('RandomForest')
    rf.fit(df_test=df_test.copy(),
           df_team_strength=df_team_strength.copy(),
           general_league_data=general_league_data.copy(),
           features=features)
    rf.predict()
    accuracy.append(rf.accuracy)
    f1.append(rf.f1_score)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Poisson Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    po = models.get_model('Poisson')
    po.predict_all(df_test=df_test.copy(),
                   df_team_strength=df_team_strength.copy(),
                   general_league_data=general_league_data.copy())
    accuracy.append(po.accuracy)
    f1.append(po.f1_score)
    # ~~~~~~~~~~~~~~~~~~~ Support Vector Machine Model ~~~~~~~~~~~~~~~~~~~~~~~#
    svm = models.get_model('SVM')
    svm.fit(df_test=df_test.copy(),
            df_team_strength=df_team_strength.copy(),
            general_league_data=general_league_data.copy(),
            features=features)
    svm.predict()
    accuracy.append(svm.accuracy)
    f1.append(svm.f1_score)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~ XGBoost Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    xgb = models.get_model('XGBoost')
    xgb.fit(df_test=df_test.copy(),
            df_team_strength=df_team_strength.copy(),
            general_league_data=general_league_data.copy(),
            features=features)
    xgb.predict()
    accuracy.append(xgb.accuracy)
    f1.append(xgb.f1_score)
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Logistic Regression Model ~~~~~~~~~~~~~~~~~~~~~#
    logreg = models.get_model('LogisticRegression')
    logreg.fit(df_test=df_test.copy(),
               df_team_strength=df_team_strength.copy(),
               general_league_data=general_league_data.copy(),
               features=features)
    logreg.predict()

