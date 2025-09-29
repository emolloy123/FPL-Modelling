import pandas as pd
from .TeamOptimizer import TeamOptimizer

class MostSelectedTeam:
    def __init__(self, players_hist_merged: pd.DataFrame, gameweek: int):
        self.gameweek = gameweek
        self.p_hist_gameweek = players_hist_merged[
            players_hist_merged['round'] == gameweek
        ]
        self.result = None

    def get_most_selected_team(self, objective_col: str = 'selected'):
        optimizer = TeamOptimizer(self.p_hist_gameweek, objective_col)
        self.result = optimizer.solve()
        return self.result

    def get_selected_team_points(self) -> int:

        squad_df = self.p_hist_gameweek[
            self.p_hist_gameweek['player_name'].isin(self.result['squad'])
        ]
        captain = self.result['captain']
        vice = self.result['vice_captain']

        base_points = squad_df['round_points'].sum()

        # Captain doubles
        cap_points = squad_df.loc[
            squad_df['player_name'] == captain, 'round_points'
        ].sum()
        base_points += cap_points

        # Vice steps in only if captain scores 0
        if cap_points == 0:
            vice_points = squad_df.loc[
                squad_df['player_name'] == vice, 'round_points'
            ].sum()
            base_points += vice_points

        return int(base_points)
