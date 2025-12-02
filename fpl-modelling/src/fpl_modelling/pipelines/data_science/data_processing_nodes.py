import pandas as pd
import typing as tp 

def preprocess_data(df: pd.DataFrame, model_config: tp.Dict, model_num: int):
    """
    Do some intial filters on the data
    """

    df[f"next_week_round_points"] = df.groupby('player_id')['round_points'].shift(-1)

    # Remove players with no minutes
    minute_threshold = model_config[model_num]['minute_threshold']
    players_hist_merged_clean = df[df['cumsum_minutes']>minute_threshold]    

    return players_hist_merged_clean