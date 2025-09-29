from kedro.pipeline import Pipeline, node, pipeline
from .create_player_table_nodes import (
    init_api_client,
    process_players_data,
    process_teams_data,
    process_positions_data,
)


def create_players_table_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Fetch raw data from APIs
        node(
            func=init_api_client,
            inputs="params:base_url",
            outputs="client",
            name="fetch_client_node"
        ),
        # Process and store reference data in SQLite
        node(
            func=process_players_data,
            inputs=dict(
                client="client"
                ),
            outputs="players",  # This goes to SQLite
            name="process_players_data_node"
        ),
        node(
            func=process_teams_data,
            inputs=dict(client="client"),
            outputs="teams",  # This goes to SQLite
            name="process_teams_data_node"
        ),
        node(
            func=process_positions_data,
            inputs=dict(
                client="client"
                ),
            outputs="positions",  # This goes to SQLite
            name="process_positions_data_node"
        )
    ])