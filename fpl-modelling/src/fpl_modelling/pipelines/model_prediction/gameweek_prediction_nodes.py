import pandas as pd 
import typing as tp  
import logging
logger = logging.getLogger(__name__)
from fpl_modelling.ModelConfig import load_model_config
import numpy as np 
from fpl_modelling.pipelines.optimisation.TeamOptimizer import TeamOptimizer


def model_prediction_train_test(pipeline, X_train, X_test):

    return pipeline.predict(X_train), pipeline.predict(X_test)


def join_back_predictions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray
):
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train.loc[:, 'predicted_next_round_points'] = y_pred_train
    df_test.loc[:, 'predicted_next_round_points'] = y_pred_test

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
        


