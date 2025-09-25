import requests
import pandas as pd

class Player:
    def __init__(self, player_id, merged_df):
        self.player_id = player_id
        self.merged_df = merged_df
        self.player_name = merged_df[merged_df['player_id']==player_id]['player_name'].values[0]
        print('Initialised ', self.player_name)
    

    def get_player_gameweek_history(self):
        '''get all gameweek info for a given player_id'''
        
        # send GET request to
        # https://fantasy.premierleague.com/api/element-summary/{PID}/
        base_url = 'https://fantasy.premierleague.com/api/'
        r = requests.get(
                base_url + 'element-summary/' + str(self.player_id) + '/'
        ).json()

            
        print('Gameweek history for ', self.player_name)
        
        # extract 'history' data from response into dataframe

        self.gameweek_history = pd.json_normalize(r['history'])
        