import sqlite3
import pandas as pd
import requests
from fpl_modelling.FPL_API import FPLClient

def init_api_client(base_url: str):

    return FPLClient(base_url)

def process_players_data(client) -> pd.DataFrame:
    """Extract and clean players data"""

    players = client.get_players()
    players = players.rename(columns={
        'id': 'player_id', 
        'team': 'team_id', 
        'element_type': 'position_id'
    })
    players['player_name'] = players['first_name'] + ' ' + players['second_name']
    players['now_cost'] = players['now_cost']/10

    players['points_per_game'] = players['points_per_game'].astype(float)

    return players

def process_teams_data(client) -> pd.DataFrame:
    """Extract and clean teams data"""
    teams = client.get_teams()
    return teams.rename(columns={'id': 'team_id', 'name': 'team_name'})

def process_positions_data(client) -> pd.DataFrame:
    """Extract and clean positions data"""
    positions = client.get_positions()
    positions["sub_positions_locked"] = positions["sub_positions_locked"].astype(str)

    return positions.rename(columns={
        'id': 'position_id', 
        'singular_name': 'position_name'
    })