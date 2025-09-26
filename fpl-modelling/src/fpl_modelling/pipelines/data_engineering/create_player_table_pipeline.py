from kedro.pipeline import Pipeline, node, pipeline
from .create_player_table_nodes import (
    fetch_bootstrap_data,
    process_players_data,
    process_teams_data,
    process_positions_data,
    merge_player_reference_data
)


def create_players_table_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Fetch raw data from APIs
        node(
            func=fetch_bootstrap_data,
            inputs=dict(bootstrap_url="params:bootstrap_url"),
            outputs="raw_bootstrap_data",
            name="fetch_bootstrap_data_node"
        ),
        
        # Process and store reference data in SQLite
        node(
            func=process_players_data,
            inputs=dict(bootstrap_data="raw_bootstrap_data"),
            outputs="db_players",  # This goes to SQLite
            name="process_players_data_node"
        ),
        node(
            func=process_teams_data,
            inputs=dict(bootstrap_data="raw_bootstrap_data"),
            outputs="db_teams",  # This goes to SQLite
            name="process_teams_data_node"
        ),
        node(
            func=process_positions_data,
            inputs=dict(bootstrap_data="raw_bootstrap_data"),
            outputs="db_positions",  # This goes to SQLite
            name="process_positions_data_node"
        )
    ])