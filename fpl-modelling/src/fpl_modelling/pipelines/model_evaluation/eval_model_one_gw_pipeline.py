from kedro.pipeline import Pipeline, node, pipeline
from .eval_model_one_gw import regression_metrics, eval_predicted_optimal_team, log_optimal_team_metrics

def create_eval_model_one_gw_pipeline(**kwargs) -> Pipeline:

    return pipeline([        
        
        node(
            func=regression_metrics,
            inputs=dict(
                y_train="y_train",
                y_pred_test="y_pred_test",
                y_test="y_test",
                y_pred_train="y_pred_train",
                predicting_gameweek="params:predicting_gameweek",
                mlflow_run_id = "mlflow_run_id"
            ),
            outputs="regression_metrics",
            name="regression_metrics_node",
        ),
        node(
            func=eval_predicted_optimal_team,
            inputs=dict(
                players_hist_merged="players_hist_merged",
                optimiser_res="predicted_optimal_team_next_gw",
                predicting_gameweek="params:predicting_gameweek",
                
            ),
            outputs="gw_starters_data",
            name="eval_predicted_optimal_team_node",
        ),
         node(
            func=log_optimal_team_metrics,
            inputs=dict(
                gw_starters_data="gw_starters_data",
                predicting_gameweek="params:predicting_gameweek",
                average_points = "params:average_points",
                mlflow_run_id = "mlflow_run_id",
            ),
            outputs="nothing",
            name="log_optimal_team_metrics_node",
        ),
    ])