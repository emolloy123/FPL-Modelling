import pandas as pd 
import sklearn 
import typing as tp 

def points_prediction(player_info_at_gameweek: pd.DataFrame, trained_pipeline: sklearn.pipeline.Pipeline, model_config: tp.Dict, model_num: int):
    """
    Predict expected points for all players in the specified gameweek
    """

    features = model_config[model_num]['features']['num_features'] + model_config[model_num]['features']['cat_features']

    X = player_info_at_gameweek[features]
    print(trained_pipeline.predict(X))

    return trained_pipeline.predict(X)

def team_selection():
    """
    Using the predicted points select a team that maximises predicted points while adhering to FPL rules
    """
    pass