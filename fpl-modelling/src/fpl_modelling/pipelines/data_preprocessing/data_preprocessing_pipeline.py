from kedro.pipeline import Pipeline, node, pipeline
from .data_processing_nodes import filter_players_training_data, eng_rolling_avg_features, train_test_split_by_gw

def create_data_preprocessing_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict. 
    """
    return pipeline([
        node(
            func=filter_players_training_data,
            inputs=dict(
                players_hist_merged="players_hist_merged",
                model_config="params:model_config",
                model_num = "params:model_num"
            ),
            outputs="df_filtered",
            name="filter_players_training_data_node",
        ),
        node(
            func=eng_rolling_avg_features,
            inputs=dict(
                players_hist_merged="df_filtered",
                rolling_features="params:rolling_features",
            ),
            outputs="df_processed",
            name="eng_rolling_avg_features_node",
        ),
        node(
            func=train_test_split_by_gw,
            inputs=dict(
                df="df_processed",
                model_config="params:model_config",
                model_num = "params:model_num",
                predicting_gameweek = "params:predicting_gameweek"
            ),
            outputs=["X_train", "y_train", "X_test", "y_test", "df_test", "df_train"],
            name="train_test_split_by_gw_node",
        )
    ])
