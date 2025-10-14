from kedro.pipeline import Pipeline, node, pipeline
from .train_model_nodes import load_config, train_test_split, train_model


def create_train_model_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict.
    """
    return pipeline([
        node(
            func=load_config,
            inputs=dict(
                model_config = "params:model_config",
                model_num = "params:model_num"
            ),  # model_config is passed as dict param
            outputs=["pipeline", "features"],
            name="load_model_pipeline",
        ),
        node(
            func=train_test_split,
            inputs=dict(
                expanded_df="expanded_df",  # dataset name in catalog
                num_test_gameweeks="params:num_test_gameweeks"
            ),
            outputs=["train_df", "test_df"],
            name="split_train_test",
        ),
        node(
            func=train_model,
            inputs=dict(
                train_df="train_df",
                pipeline="pipeline",
                features="features",
                target_col="params:target_col"
            ),
            outputs="trained_pipeline",
            name="train_model_node",
        ),
    ])
