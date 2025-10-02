import requests
import pandas as pd

class FPLClient:
    def __init__(self, base_url="https://fantasy.premierleague.com/api/"):
        self.base_url = base_url
        self._metadata = None  # private cache

    def get_metadata(self, force_refresh=False):
        """Fetch and cache FPL bootstrap-static data (general game metadata)."""
        if self._metadata is None or force_refresh:
            endpoint = "bootstrap-static"
            self._metadata = requests.get(self.base_url + endpoint).json()
        return self._metadata

    def get_players(self):
        """Return current players dataframe."""
        r = self.get_metadata()
        return pd.json_normalize(r['elements'])
    
    def get_teams(self):
        """Return current teams dataframe."""
        r = self.get_metadata()
        return pd.json_normalize(r['teams'])

    def get_positions(self):
        r = self.get_metadata()
        return pd.json_normalize(r['element_types'])

    def get_players_hist(self, player_id):

        endpoint = f"{self.base_url}element-summary/{player_id}/"

        r = requests.get(endpoint).json()

        return pd.json_normalize(r['history'])

    def get_gameweek_fixtures(self, gameweek_number, keep_stats=False):
        endpoint = f"{self.base_url}fixtures/?event={gameweek_number}"
        r = requests.get(endpoint).json()   # r is a list of dicts (one per fixture)

        if not keep_stats:
            for fixture in r:
                fixture.pop("stats", None)  # safely remove "stats" if present

        return pd.json_normalize(r)

    def get_team_on_gameweek(self, team_id: int, gameweek: int):

        endpoint = f"{self.base_url}entry/{team_id}/event/{gameweek}/picks/"

        r = requests.get(endpoint).json()   
        
        gameweek_team = pd.json_normalize(r['picks'])
        gameweek_summary = r['entry_history']

        return {
            "gameweek_team": gameweek_team,
            "gameweek_summary": gameweek_summary
        }

    def get_team_transfers(self, team_id: int):

        endpoint = f"{self.base_url}entry/{team_id}/transfers/"

        r = requests.get(endpoint).json()  

        return pd.DataFrame(r)
    
    def get_current_team(self, team_id: int, bearer: str):

        endpoint = f"{self.base_url}my-team/{team_id}"
        r = requests.get(endpoint, headers={ "X-Api-Authorization": f"Bearer {bearer}"})

        return r.json()