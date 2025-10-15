from fpl_modelling.FPL_API import FPLClient
import pandas as pd 
from .create_fixtures_table_nodes import get_fixtures
from kedro.pipeline import Pipeline, node

def create_fixtures_table_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_fixtures,
                inputs=None,
                outputs="fixtures",
                name="create_fixtures_table_node",
            ),
        ]
    )
