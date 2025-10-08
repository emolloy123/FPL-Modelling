"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from fpl_modelling.pipelines.data_engineering.create_player_table_pipeline import create_players_table_pipeline
from fpl_modelling.pipelines.optimisation.pick_team_pipelines import (
    create_pick_optimal_team_pipeline, 
    create_pick_most_selected_team_pipeline
)
from fpl_modelling.pipelines.data_engineering.create_player_gw_hist_table_pipeline import create_player_gw_hist_table_pipeline
from fpl_modelling.pipelines.data_engineering.get_team_pipeline import create_get_team_pipeline
def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    players_table_pipeline = create_players_table_pipeline()

    pick_optimal_team_pipeline = create_pick_optimal_team_pipeline()

    pick_most_selected_team_pipeline = create_pick_most_selected_team_pipeline()

    player_gw_hist_table_pipeline = create_player_gw_hist_table_pipeline()

    get_team_pipeline = create_get_team_pipeline()


    return {
        "create_players_table": players_table_pipeline,
        "pick_optimal_team": pick_optimal_team_pipeline,
        "create_player_gw_hist_table": player_gw_hist_table_pipeline,
        "pick_most_selected_team": pick_most_selected_team_pipeline,
        "get_team": get_team_pipeline,
        "update_tables": players_table_pipeline + player_gw_hist_table_pipeline
    }
