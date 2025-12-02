import pandas as pd
import typing as tp 
from .ExpandingDF import ExpandingDF


def init_transformations(players_hist_merged_orig: pd.DataFrame, model_config: tp.Dict, model_num: int):
    """
    Do some intial filters on the data
    """

    players_hist_merged_orig[f"next_week_round_points"] = players_hist_merged_orig.groupby('player_id')['round_points'].shift(-1)

    # Remove players with no minutes
    minute_threshold = model_config[model_num]['minute_threshold']
    players_hist_merged_clean = players_hist_merged_orig[players_hist_merged_orig['total_minutes']>minute_threshold]    

    return players_hist_merged_clean

# def expand_df(current_gameweek:int, players_hist_merged_clean: pd.DataFrame, num_games_to_predict: int, target_col:str, num_test_gameweeks: int):
#     """
#     Expand df to stack examples per player by gameweek
#     """

#     expander = ExpandingDF(current_gameweek=current_gameweek,  arg_df=players_hist_merged_clean, num_games_to_predict=num_games_to_predict,
#                             target_col=target_col)
    
#     expanded_df = expander.expand_df()

#     return expanded_df

