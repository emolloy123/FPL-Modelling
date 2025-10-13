import pandas as pd 

class ExpandingDF:
    def __init__(self, current_gameweek: int,  arg_df, num_forecasted_games: int=3, target_col: str='next_week_round_points', num_test_gameweeks: int=1):

        self.current_gameweek = current_gameweek
        self.arg_df = arg_df
        self.num_forecasted_games = num_forecasted_games
        self.target_col = target_col
        self.num_test_gameweeks = num_test_gameweeks

        self._check_data()

    def _check_data(self):

        max_gw_in_data = self.arg_df['round'].max()+1

        # --- Sanity checks ---
        if self.current_gameweek > max_gw_in_data:
            raise ValueError(f"current_gameweek={self.current_gameweek} is beyond max round in data ({max_gw_in_data}).")

        if self.current_gameweek + self.num_forecasted_games > max_gw_in_data:
            raise ValueError(
                f"Not enough future gameweeks after GW{self.current_gameweek} "
                f"to forecast {self.num_forecasted_games} games. "
                f"Max available GW is {max_gw_in_data}."
            )

    def expand_df(self):

        """
        Loop backwards from current gameweek setting target as total points equal to the 3 gameweeks after it
        """
        Xs = []

        # Go backwards from current_gameweek down to 1
        for gw in range(self.current_gameweek, 0, -1):
            curr_X = self.arg_df[self.arg_df['round'] == gw].copy()

            # Define next gameweeks to sum for target
            next_weeks = list(range(gw + 1, gw + 1 + self.num_forecasted_games))

            target_data = self.arg_df[self.arg_df['round'].isin(next_weeks)][['player_id', self.target_col]]

            # Sum future points per player
            curr_y = target_data.groupby('player_id')[self.target_col].sum().reset_index(name='target')

            # Merge features and target
            merged = pd.merge(curr_X, curr_y, on='player_id', how='inner')
            Xs.append(merged)

        # Combine all GWs
        expanded_df = pd.concat(Xs, ignore_index=True)

        return expanded_df

    def gw_train_test_split(self, expanded_df, num_test_gameweeks = 1):
        """
        Set the 'num_test_gameweeks' number of gameweeks as test set, setting most recent gameweeks as test
        """

        # Split into train/test
        max_gw = expanded_df['round'].max()
        test_gws = list(range(max_gw, max_gw - num_test_gameweeks, -1))

        test_df = expanded_df[expanded_df['round'].isin(test_gws)].reset_index(drop=True)
        train_df = expanded_df[~expanded_df['round'].isin(test_gws)].reset_index(drop=True)

        return train_df, test_df