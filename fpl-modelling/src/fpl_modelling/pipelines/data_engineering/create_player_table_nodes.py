import sqlite3
import pandas as pd
import requests

def fetch_bootstrap_data(bootstrap_url: str) -> dict:
    """Fetch raw bootstrap data from API"""
    return requests.get(bootstrap_url).json()

def process_players_data(bootstrap_data: dict) -> pd.DataFrame:
    """Extract and clean players data"""
    players = pd.json_normalize(bootstrap_data['elements'])
    players = players.rename(columns={
        'id': 'player_id', 
        'team': 'team_id', 
        'element_type': 'position_id'
    })
    players['player_name'] = players['first_name'] + ' ' + players['second_name']
    return players

def process_teams_data(bootstrap_data: dict) -> pd.DataFrame:
    """Extract and clean teams data"""
    teams = pd.json_normalize(bootstrap_data['teams'])
    return teams.rename(columns={'id': 'team_id', 'name': 'team_name'})

def process_positions_data(bootstrap_data: dict) -> pd.DataFrame:
    """Extract and clean positions data"""
    positions = pd.json_normalize(bootstrap_data['element_types'])
    positions["sub_positions_locked"] = positions["sub_positions_locked"].astype(str)

    return positions.rename(columns={
        'id': 'position_id', 
        'singular_name': 'position_name'
    })

def merge_player_reference_data(
    players: pd.DataFrame, 
    teams: pd.DataFrame, 
    positions: pd.DataFrame
) -> pd.DataFrame:
    """Merge players with teams and positions reference data"""
    df = players.merge(teams, on='team_id')
    df = df.merge(positions, on='position_id')
    return df

# def

# price_changes = [
#     ['28/8', 'Raya', 'Arsenal', 5.5, 5.6],
#     ['28/8', 'Chalobah', 'Chelsea', 5.1, 5.2],
#     ['28/8', 'McAtee', "Nott'm Forest", 5.4, 5.3]
# ]

# # Convert to DataFrame
# df = pd.DataFrame(price_changes, columns=["date", "player", "team", "old_price", "new_price"])

# # Connect to SQLite DB (creates file if it doesn’t exist)
# conn = sqlite3.connect("fpl_data.db")

# # Write to DB
# df.to_sql("price_changes", conn, if_exists="replace", index=False)

# # Query example
# query = "SELECT * FROM price_changes WHERE date='28/8'"
# results = pd.read_sql(query, conn)
# print(results)

# conn.close()
