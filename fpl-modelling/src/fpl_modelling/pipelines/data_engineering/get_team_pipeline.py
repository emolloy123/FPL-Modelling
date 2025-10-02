
from kedro.pipeline import Pipeline, node

from .get_team_nodes import get_current_team
def create_get_team_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_current_team,
                inputs=dict(
                    base_url="params:base_url",
                    bearer = "params:bearer",
                    team_id = "params:my_team_id"
                    ),
                outputs="my_current_team",
                name="get_current_team_node",
            ),
        ]
    )
