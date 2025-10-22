import pandas as pd 
import sklearn 
import typing as tp 
import mlflow 

def points_prediction(player_info_at_gameweek: pd.DataFrame, model_config: tp.Dict, model_num: int):
    """
    Predict expected points for all players in the specified gameweek
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    trained_pipeline= mlflow.sklearn.load_model("models:/fpl_model/latest")

    features = model_config[model_num]['features']['num_features'] + model_config[model_num]['features']['cat_features']

    X = player_info_at_gameweek[features]
    y_pred = trained_pipeline.predict(X)
    players_df = player_info_at_gameweek.copy()

    # Add or replace the points column with model predictions
    players_df['points_per_game'] = y_pred
    players_df['now_cost'] = 10

    players_df = players_df.rename(columns={
        "players_team": "team_id",
        # "position_name": "position"
    })
    return players_df

