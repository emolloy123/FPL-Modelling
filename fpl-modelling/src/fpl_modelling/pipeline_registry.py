"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from fpl_modelling.pipelines.data_engineering.create_player_table_pipeline import create_players_table_pipeline



def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    players_table_pipeline = create_players_table_pipeline()


    return {
        "create_players_table": players_table_pipeline
    }
