from .gameweek_prediction_nodes import predict_points_every_player_next_gameweek, get_predicted_optimal_team_next_gameweek

from kedro.pipeline import Pipeline, node, pipeline


def create_gameweek_prediction_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict.
    """
    return pipeline([
        node(
            func=predict_points_every_player_next_gameweek,
            inputs=dict(
                pipeline = "pipeline",
                model_config = "params:model_config",
                X_test = "X_test"
            ),  
            outputs="players_hist_merged_w_predictions",
            name="predict_points_every_player_next_gameweek_node",
        ),
        node(
            func=get_predicted_optimal_team_next_gameweek,
            inputs=dict(
                df_with_predictions = "players_hist_merged_w_predictions",
            ),  
            outputs="predicted_optimal_team_next_gw",
            name="get_predicted_optimal_team_next_gameweek_node",
        ),
    ])
    
