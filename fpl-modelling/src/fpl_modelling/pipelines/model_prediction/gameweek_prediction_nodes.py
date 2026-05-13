import pandas as pd 
import sklearn 
import typing as tp  
import mlflow 
import logging
logger = logging.getLogger(__name__)
from fpl_modelling.ModelConfig import load_model_config
import numpy as np 
from fpl_modelling.pipelines.optimisation.TeamOptimizer import TeamOptimizer


def model_prediction_train_test(pipeline, X_train, X_test):

    return pipeline.predict(X_train), pipeline.predict(X_test)


def join_back_predictions(df_train: pd.DataFrame, df_test: pd.DataFrame, y_pred_train: np.ndarray, y_pred_test: np.ndarray):

    df_train['predicted_next_round_points'] = y_pred_train

    df_test['predicted_next_round_points'] = y_pred_test

    return df_train, df_test

def get_predicted_optimal_team_next_gameweek(df_test: pd.DataFrame, objective_col = 'predicted_next_round_points'):

    # Aggregate to player level (handles DGWs correctly)

    df_test = (
            df_test
            .groupby(['player_id'], as_index=False)
            .agg({
                "predicted_next_round_points": 'sum',              # predicted points
                'next_week_round_points': 'sum',         # actual points
                'transfer_cost': 'first',
                'position_name': 'first',
                'team_id': 'first',
                'player_name': 'first',
                'round': 'first'
            })
        )

    optimizer = TeamOptimizer(df_test, kpi_col="predicted_next_round_points") 

    return optimizer.solve(budget=1e6, print_sol=False)
        

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

