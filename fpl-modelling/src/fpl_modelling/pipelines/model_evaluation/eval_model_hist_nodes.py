# eval_model_hist_nodes.py

from fpl_modelling.pipelines.data_preprocessing.data_processing_nodes import (
    filter_players_training_data,
    eng_rolling_avg_features,
    train_test_split_by_gw,
    filter_players_prediction_data
)
from fpl_modelling.pipelines.model_training.train_model_nodes import train_model
from fpl_modelling.pipelines.model_prediction.gameweek_prediction_nodes import (
    model_prediction_train_test,
    join_back_predictions,
    get_predicted_optimal_team_next_gameweek,
)
from fpl_modelling.pipelines.model_evaluation.eval_model_one_gw import regression_metrics, eval_predicted_optimal_team
from fpl_modelling.ModelConfig import load_model_config
from sklearn.base import clone
import pandas as pd 
import mlflow
import typing as tp 
import plotly.graph_objects as go

import logging 
logger = logging.getLogger(__name__)
from .plotting import plot_multi_metrics, plot_metrics_by_gw

def eval_model_walk_forward(
    players_hist_merged: pd.DataFrame,
    model_config: tp.Dict,
    model_num: int,
    rolling_features: tp.Dict,
    average_points: tp.Dict,
    mlflow_tracking_uri
    
):
    base_pipeline, features = load_model_config(model_config, model_num)
    max_gameweek = int(players_hist_merged["round"].max())

    metrics_by_gw, picked_teams = {}, {}

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(f"walk_forward_cross_validation")

    with mlflow.start_run(run_name=f"model_{model_num}") as parent_run:

        for gw in range(5, max_gameweek):

            with mlflow.start_run(run_name=f"gw_{gw}", nested=True):

                logger.info(f"--- Gameweek {gw} ---")
                try:
                    # -- data_preprocessing_pipeline nodes --
                    df_visible = players_hist_merged[players_hist_merged["round"] <= gw].copy()
                    df_filtered = filter_players_training_data(df_visible, model_config, model_num)
                    df_filtered = filter_players_prediction_data(df_filtered)
                    df_processed = eng_rolling_avg_features(df_filtered, rolling_features)
                    X_train, y_train, X_test, y_test, df_test, df_train = train_test_split_by_gw(
                        model_num, df_processed, model_config, predicting_gameweek=gw
                    )

                    # -- train_model_pipeline nodes --
                    fold_pipeline = clone(base_pipeline)
                    trained_pipeline, _ = train_model(
                        X_train, y_train,
                        pipeline=fold_pipeline,
                        predicting_gameweek=gw,
                        mlflow_tracking_uri=None,
                    )
                    cat_features = model_config[model_num]['features']['cat_features']
                    logger.info(f"GW {gw} - NaNs in cat features:\n{X_test[cat_features].isna().sum()}")
                    logger.info(f"GW {gw} - unique values in cat features:\n{X_test[cat_features].apply(lambda x: x.unique())}")

                    # -- gameweek_prediction_pipeline nodes --
                    y_pred_train, y_pred_test = model_prediction_train_test(trained_pipeline, X_train, X_test)
                    df_train_w_pred, df_test_w_pred = join_back_predictions(df_train, df_test, y_pred_train, y_pred_test)
                    optimal_team = get_predicted_optimal_team_next_gameweek(df_test_w_pred)

                    # -- eval_model_one_gw_pipeline nodes --
                    metrics = regression_metrics(y_train, y_test, y_pred_test, y_pred_train, gw, mlflow_run_id=None)
                    gw_starters_data = eval_predicted_optimal_team(players_hist_merged, optimal_team, gw)


                    metrics = add_optimal_team_metrics(gw_starters_data, average_points, gw, metrics)
                    mlflow.log_metrics(metrics, step=gw)

                    metrics_by_gw[gw] = metrics

                    picked_teams[gw] = gw_starters_data

                except Exception as e:
                    logger.error(f"GW {gw} failed: {e}", exc_info=True)
                    continue

    return pd.DataFrame(metrics_by_gw).T, parent_run.info.run_id


def add_optimal_team_metrics(gw_starters_data: pd.DataFrame, average_points: tp.Dict, predicting_gameweek: int, metrics: tp.Dict):
     
    metrics['predicted_team_total_true_points'] = gw_starters_data['next_week_round_points'].sum()
    metrics['predicted_team_total_predicted_points'] = gw_starters_data['predicted_next_round_points'].sum()
    metrics['average_points_that_gw'] = average_points[predicting_gameweek]


    return metrics

def log_average_metrics(metrics_df: pd.DataFrame, mlflow_run_id):
    avg_metrics = {}
    with mlflow.start_run(run_id=mlflow_run_id):
        for col in metrics_df.columns:
            avg_metrics[f'avg_{col}'] = metrics_df[col].mean()

        
        avg_metrics['sum_predicted_team_true_points'] = metrics_df['predicted_team_total_true_points'].sum()
        avg_metrics['sum_average_points'] = metrics_df['average_points_that_gw'].sum()
        avg_metrics['sum_predicted_team_predicted_points'] = metrics_df['predicted_team_total_predicted_points'].sum()

        mlflow.log_metrics(avg_metrics)

    return ""

def add_plots(metrics_df, mlflow_run_id):

    plot_multi_metrics(metrics_df, ["predicted_team_total_true_points", "average_points_that_gw"], mlflow_run_id, title="Predicted vs Average Team" )
    plot_metrics_by_gw(metrics_df, mlflow_run_id)

    return " "






