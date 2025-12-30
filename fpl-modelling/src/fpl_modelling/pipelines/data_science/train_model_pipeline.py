from kedro.pipeline import Pipeline, node, pipeline
from .train_model_nodes import load_config, train_test_split, train_model
from .data_processing_nodes import preprocess_data, eng_rolling_avg_features

def create_train_model_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict.
    """
    return pipeline([
        node(
            func=preprocess_data,
            inputs=dict(
                df = "players_hist_merged",
                model_config = "params:model_config",
                model_num = "params:model_num"
            ),  # model_config is passed as dict param
            outputs="df_processed_imd",
            name="preprocess_data_node",
        ),  
        node(
            func=eng_rolling_avg_features,
            inputs=dict(
                df = "df_processed_imd",
                rolling_features = "params:rolling_features",
            ),  # model_config is passed as dict param
            outputs="df_processed",
            name="eng_rolling_avg_features_node",
        ),  
        
        node(
            func=load_config,
            inputs=dict(
                model_config = "params:model_config",
                model_num = "params:model_num"
            ),  # model_config is passed as dict param
            outputs=["pipeline", "features"],
            name="load_model_node",
        ),
        node(
            func=train_test_split,
            inputs=dict(
                df="df_processed",  # dataset name in catalog
                predicting_gameweek="params:predicting_gameweek"
            ),
            outputs=["train_df", "test_df"],
            name="split_train_test_node",
        ),
        node(
            func=train_model,
            inputs=dict(
                train_df="train_df",
                pipeline="pipeline",
                features="features",
                mlflow_tracking_uri = "params:mlflow_tracking_uri",
                predicting_gameweek = "params:predicting_gameweek"
            ),
            outputs="trained_pipeline",
            name="train_model_node",
        ),
    ])
