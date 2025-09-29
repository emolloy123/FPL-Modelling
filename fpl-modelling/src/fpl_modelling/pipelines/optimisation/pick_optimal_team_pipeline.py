from kedro.pipeline import Pipeline, node, pipeline
from .pick_optimal_team_nodes import (
    pick_optimal_team,
)


def create_pick_optimal_team_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Fetch raw data from APIs
        node(
            func=pick_optimal_team,
            inputs=dict(
                players_df="players_merged",
                objective_col = "params:objective_col"
                ),
            outputs="optimal_team",
            name="pick_optimal_team_node"
        ),
        
    ])