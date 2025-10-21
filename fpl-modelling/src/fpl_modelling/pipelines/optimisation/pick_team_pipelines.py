from kedro.pipeline import Pipeline, node, pipeline
from .pick_team_nodes import (
    pick_most_selected_team,
    pick_optimal_team
)

def create_pick_optimal_team_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Fetch raw data from APIs
        node(
            func=pick_optimal_team,
            inputs=dict(
                players_df="opt_input_df",
                objective_col = "params:objective_col"
                ),
            outputs="optimal_team",
            name="pick_optimal_team_node"
        ),
    ])
    
def create_pick_most_selected_team_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=pick_most_selected_team,
            inputs=dict(
                players_hist_merged="players_hist_merged",
                gameweek = "params:gameweek"
                ),
            outputs="most_selected_team",
            name="pick_most_selected_team_node"
        ),
    ])