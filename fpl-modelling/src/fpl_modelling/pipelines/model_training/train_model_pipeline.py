from kedro.pipeline import Pipeline, node, pipeline
from .train_model_nodes import train_model
from fpl_modelling.ModelConfig import load_model_config
def create_train_model_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict. 
    """
    return pipeline([       
        node(
            func=load_model_config,
            inputs=dict(
                model_config = "params:model_config",
                model_num = "params:model_num"
            ),  # model_config is passed as dict param
            outputs=["pipeline", "features"],
            name="load_model_node",
        ),
        node(
            func=train_model,
            inputs=dict(
                X_train="X_train",
                y_train = "y_train",
                pipeline="pipeline",
                mlflow_tracking_uri = "params:mlflow_tracking_uri",
                predicting_gameweek = "params:predicting_gameweek"
            ),
            outputs="trained_pipeline",
            name="train_model_node",
        ),
    ])
