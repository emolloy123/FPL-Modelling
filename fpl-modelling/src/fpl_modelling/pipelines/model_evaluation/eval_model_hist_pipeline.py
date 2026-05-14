"""
Kedro pipeline for evaluating FPL prediction models on historical data.

This pipeline evaluates a model's performance across multiple gameweeks by:
1. Training on all previous gameweeks
2. Predicting the next gameweek
3. Calculating performance metrics
4. Selecting optimal teams based on predictions
"""

from kedro.pipeline import Pipeline, node, pipeline
from .eval_model_hist_nodes import eval_model_walk_forward, log_average_metrics, add_plots

def create_eval_model_hist_pipeline(**kwargs) -> Pipeline:
    """
    """
    return pipeline([        
        
        node(
            func=eval_model_walk_forward,
            inputs=dict(
                players_hist_merged="players_hist_merged",
                model_config="params:model_config",
                model_num="params:model_num",
                rolling_features = "params:rolling_features",
                average_points = "params:average_points",
                mlflow_tracking_uri = "params:mlflow_tracking_uri"
            ),
            outputs=["metrics_by_gw", "mlflow_run_id"],
            name="eval_model_node",
        ),

        node(
            func=add_plots,
            inputs=dict(
                metrics_df="metrics_by_gw",
                mlflow_run_id = "mlflow_run_id"
            ),
            outputs="nothing",
            name="plot_metrics_by_gw_node",
        ),
         node(
            func=log_average_metrics,
            inputs=dict(
                metrics_df="metrics_by_gw",
                mlflow_run_id = "mlflow_run_id"
            ),
            outputs="also_nothing",
            name="log_average_metrics_node",
        ),
    ])