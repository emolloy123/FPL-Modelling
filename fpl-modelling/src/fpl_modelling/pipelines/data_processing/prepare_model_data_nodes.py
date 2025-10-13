import pandas as pd
import typing as tp 
from .ExpandingDF import ExpandingDF


def init_transformations(players_hist_merged_orig: pd.DataFrame, shift_to_next_week_cols: tp.List[str], minute_threshold: float):
    """
    Do some intial filters on the data
    """

    for shift_to_next_week_col in shift_to_next_week_cols:

        players_hist_merged_orig[f"next_week_{shift_to_next_week_col}"] = players_hist_merged_orig.groupby('player_id')[shift_to_next_week_col].shift(-1)

    # Accumulate minutes by round
    players_hist_merged_orig['accumulated_minutes'] = players_hist_merged_orig.groupby('player_id')['round_minutes'].cumsum()

    # Remove players with no minutes
    players_hist_merged_clean = players_hist_merged_orig[players_hist_merged_orig['total_minutes']>minute_threshold]    

    return players_hist_merged_clean

def expand_df(current_gameweek:int, players_hist_merged_clean:pd.DataFrame, num_forecasted_games: int, target_col:str, num_test_gameweeks: int):

    expander = ExpandingDF(current_gameweek=current_gameweek,  arg_df=players_hist_merged_clean, num_forecasted_games=num_forecasted_games,
                            target_col=target_col)
    
    expanded_df = expander.expand_df()

    train_df, test_df = expander.gw_train_test_split(expanded_df=expanded_df, num_test_gameweeks=num_test_gameweeks)