import pandas as pd 
import sklearn 
import typing as tp  
import mlflow 
import logging
logger = logging.getLogger(__name__)
from fpl_modelling.ModelConfig import load_model_config
import numpy as np 
from fpl_modelling.pipelines.optimisation.pick_team_nodes import pick_optimal_team

def predict_points_every_player_next_gameweek(model_config: tp.Dict, model_num: int, df: pd.DataFrame, predicting_gameweek):

    
    pipeline, features = load_model_config(model_config, model_num)

    X = df[df['round']==predicting_gameweek][features]

    return pipeline.predict(X)


def join_back_predictions(model_predictions: np.ndarray, players_hist_merged):

    players_hist_merged['predicted_next_week_roud_points'] = model_predictions

    return players_hist_merged

def get_predicted_optimal_team_next_gameweek(df_with_predictions, objective_col = 'predicted_next_week_roud_points'):

    # Aggregate to player level (handles DGWs correctly)

    df_with_predictions = (
            df_with_predictions
            .groupby(['player_id'], as_index=False)
            .agg({
                "predicted_next_week_points": 'sum',              # predicted points
                'true_next_week_points': 'sum',         # actual points
                'transfer_cost': 'first',
                'position_name': 'first',
                'team_id': 'first',
                'player_name': 'first',
                'round': 'first'
            })
        )
        
    return pick_optimal_team(df_with_predictions, objective_col=objective_col, print_sol=False)


# def points_prediction(df: pd.DataFrame, model_config: tp.Dict, model_num: int, mlflow_tracking_uri: str, predicting_gameweek: int, trained_pipeline=None):
#     """
#     Predict expected points for all players in the specified gameweek
#     """
#     predicting_gameweek = predicting_gameweek-1
#     mlflow.set_tracking_uri(mlflow_tracking_uri)
#     if trained_pipeline is None:
#         trained_pipeline= mlflow.sklearn.load_model(f"models:/model_gameweek_{predicting_gameweek}/latest")

#     features = model_config[model_num]['features']['num_features'] + model_config[model_num]['features']['cat_features']

    
#     X = df[df['round']==predicting_gameweek][features]
#     y_pred = trained_pipeline.predict(X)
#     players_df = df[df['round']==predicting_gameweek].copy()

#     # Add or replace the points column with model predictions
#     players_df['predicted_next_week_points'] = y_pred

#     return players_df

