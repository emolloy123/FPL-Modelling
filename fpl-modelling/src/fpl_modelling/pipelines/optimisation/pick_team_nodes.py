from .TeamOptimizer import TeamOptimizer
import pandas as pd 
import numpy as np 
from .MostSelectedTeam import MostSelectedTeam
# STRAT 1 objective_col: points_per_game
def pick_optimal_predicted_team_next_gw(players_df: pd.DataFrame, objective_col: str, print_sol=True):


    sub_df = players_df
    optimizer = TeamOptimizer(sub_df, kpi_col="predicted_next_round_points") 

    res = optimizer.solve(budget=1e6, print_sol=print_sol)

    return res

def pick_most_selected_team(players_hist_merged: pd.DataFrame, gameweek:int, objective_col = 'selected'):

    team_picker = MostSelectedTeam(players_hist_merged, gameweek)

    team_picker.get_most_selected_team()

    print('Team points:',team_picker.get_selected_team_points())

    return 2
