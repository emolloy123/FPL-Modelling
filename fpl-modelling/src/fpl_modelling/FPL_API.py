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
