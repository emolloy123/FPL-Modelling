
from kedro.pipeline import Pipeline, node

from .create_player_gw_hist_table_nodes import create_player_gw_hist_table
def create_player_gw_hist_table_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_player_gw_hist_table,
                inputs=dict(
                    db_players="players"
                    ),
                outputs="players_hist",
                name="create_players_gw_hist_node",
            ),
        ]
    )
