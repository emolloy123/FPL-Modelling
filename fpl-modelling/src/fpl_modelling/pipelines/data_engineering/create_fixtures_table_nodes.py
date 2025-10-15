from fpl_modelling.FPL_API import FPLClient
import pandas as pd 

def get_fixtures() -> pd.DataFrame:
    """Fetch gameweek fixtures for all gameweeks."""

    client = FPLClient()
    df_list = []

    for gw in range(1, 39):
        fixtures_df = client.get_gameweek_fixtures(gw)
        fixtures_df["gameweek"] = gw  # optional but often useful
        df_list.append(fixtures_df)

    return pd.concat(df_list, ignore_index=True)