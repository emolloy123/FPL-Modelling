from fpl_modelling.FPL_API import FPLClient
import pandas as pd 

def get_current_team(base_url: str, team_id: int, bearer: str):

    client = FPLClient(base_url)

    res = client.get_current_team(team_id, bearer)

    team = pd.DataFrame(res['picks'])
    
    team = team.rename(columns={
    'element': 'player_id',
    'element_type': 'position_id',
    })
    print(team)
    return team

