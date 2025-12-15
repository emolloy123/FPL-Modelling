from .eval_model_hist_nodes import *


from kedro.pipeline import Pipeline, node, pipeline


def create_eval_model_hist_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict.
    """
    return pipeline([
        node(
            func=eval_model,
            inputs=dict(
                players_hist_merged = "players_hist_merged",
                model_config = "params:model_config",
                model_num = "params:model_num",
            ),  
            outputs="metrics",
            name="eval_model_node",
        ),
    ])