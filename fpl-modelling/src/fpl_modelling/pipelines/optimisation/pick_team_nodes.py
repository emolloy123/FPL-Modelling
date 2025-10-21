from .TeamOptimizer import TeamOptimizer
import pandas as pd 
import numpy as np 
from .MostSelectedTeam import MostSelectedTeam
# STRAT 1 objective_col: points_per_game
def pick_optimal_team(players_df: pd.DataFrame, objective_col: str='points'):

    # sub_df = players_df[players_df['total_minutes']>180]

    # # Ensure player is not injured or anything
    # sub_df = sub_df[sub_df['chance_of_playing_next_round'].isin([np.nan, 100])]

    # print(list(sub_df.columns))
    sub_df = players_df
    print(sub_df)
    optimizer = TeamOptimizer(sub_df, objective_col) 

    res = optimizer.solve( budget=1e6)

    return res

def pick_most_selected_team(players_hist_merged: pd.DataFrame, gameweek:int, objective_col = 'selected'):

    team_picker = MostSelectedTeam(players_hist_merged, gameweek)

    team_picker.get_most_selected_team()

    print('Team points:',team_picker.get_selected_team_points())

    return 2
