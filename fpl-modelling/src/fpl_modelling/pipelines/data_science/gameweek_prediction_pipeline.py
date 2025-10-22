from .gameweek_prediction_nodes import points_prediction

from kedro.pipeline import Pipeline, node, pipeline


def create_gameweek_prediction_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict.
    """
    return pipeline([
        node(
            func=points_prediction,
            inputs=dict(
                player_info_at_gameweek = "player_info_at_gameweek",
                # trained_pipeline = "trained_pipeline",
                model_config = "params:model_config",
                model_num = "params:model_num"
            ),  
            outputs="opt_input_df",
            name="points_prediction_pipeline",
        ),
    ])
