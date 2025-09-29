from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests


def fetch_player(row, gameweek_history_url: str) -> pd.DataFrame:
    """Fetch gameweek history for a single player."""
    url = gameweek_history_url.format(player_id=row.Index)
    r = requests.get(url).json()

    # extract 'history' data from response into dataframe
    player_df = pd.json_normalize(r["history"])
    player_df = player_df.rename(columns={'id':'player_id'})
    player_df["player_id"] = row.Index
    player_df["player_name"] = row.player_name
    player_df['team_id'] = row.team_id

    return player_df


def create_player_gw_hist_table(
    db_players: pd.DataFrame, gameweek_history_url: str
) -> pd.DataFrame:
    """Fetch gameweek history for all players and combine into one dataframe."""
    df_list = []

    with ThreadPoolExecutor(max_workers=20) as executor:  # tune worker count
        futures = [
            executor.submit(fetch_player, row, gameweek_history_url)
            for row in db_players.itertuples(index=True)
        ]
        for f in as_completed(futures):
            df_list.append(f.result())

    player_gw_hist = pd.concat(df_list, ignore_index=True)
    print(list(player_gw_hist.columns))
    return player_gw_hist
