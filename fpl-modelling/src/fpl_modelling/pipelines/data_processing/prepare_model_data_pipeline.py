from kedro.pipeline import Pipeline, node, pipeline
from .prepare_model_data_nodes import init_transformations, expand_df


def create_prepare_model_data_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Fetch raw data from APIs
        node(
            func=init_transformations,
            inputs = dict(
                players_hist_merged_orig = "players_hist_merged",
                shift_to_next_week_cols = "params:shift_to_next_week_cols",
                minute_threshold = "globals:minute_threshold"
            ),
            outputs="players_hist_merged_clean",
            name="finit_transformations_node"
        ),
        # Process and store reference data in SQLite
        node(
            func=expand_df,
            inputs=dict(
                current_gameweek = "params:current_gameweek",
                players_hist_merged_clean="players_hist_merged_clean",
                num_games_to_predict = "params:num_games_to_predict",
                target_col = "params:target_col",
                num_test_gameweeks = "params:num_test_gameweeks"
                ),
            outputs="expanded_df",  # This goes to SQLite
            name="expand_df_node"
        )
    ])