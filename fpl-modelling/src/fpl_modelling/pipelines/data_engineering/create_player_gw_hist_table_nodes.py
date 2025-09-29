from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from fpl_modelling.FPL_API import FPLClient

def fetch_player(row, client) -> pd.DataFrame:
    """Fetch gameweek history for a single player."""


    # extract 'history' data from response into dataframe
    player_df = client.get_players_hist(row.Index)
    player_df = player_df.rename(columns={'id':'player_id'})
    player_df["player_id"] = row.Index
    player_df["player_name"] = row.player_name
    player_df['team_id'] = row.team_id
    player_df['position_id'] = row.position_id

    return player_df


def create_player_gw_hist_table(
    db_players: pd.DataFrame
) -> pd.DataFrame:
    """Fetch gameweek history for all players and combine into one dataframe."""
    df_list = []

    with ThreadPoolExecutor(max_workers=20) as executor:  # tune worker count
        futures = [
            executor.submit(fetch_player, row,  FPLClient())
            for row in db_players.itertuples(index=True)
        ]
        for f in as_completed(futures):
            df_list.append(f.result())

    player_gw_hist = pd.concat(df_list, ignore_index=True)
    
    return player_gw_hist
