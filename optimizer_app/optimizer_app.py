import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="FPL Team Optimizer", layout="wide")

st.title("⚽ FPL Wildcard Team Optimizer")
st.markdown(
    "Pick your best team for the next gameweek using **model predictions** and **optimization**. "
    "*(Fake data for now)*"
)

# --- Fake player data ---
def load_player_data():
    data = {
        "player_name": [f"Player {i}" for i in range(1, 21)],
        "team": ["Team A", "Team B", "Team C", "Team D"] * 5,
        "position": ["GK", "DEF", "MID", "FWD"] * 5,
        "predicted_points": np.round(np.random.uniform(2, 10, 20), 1),
    }
    return pd.DataFrame(data)

players_df = load_player_data()

# --- Fake optimization ---
def select_best_team(df):
    df = df.copy()
    # starters + bench
    starters = []
    bench = []
    positions = ["GK", "DEF", "MID", "FWD"]
    n_starters = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
    n_bench = {"GK": 1, "DEF": 1, "MID": 1, "FWD": 1}
    
    for pos in positions:
        pos_players = df[df["position"] == pos].sample(n=n_starters[pos] + n_bench[pos])
        starters.append(pos_players.iloc[:n_starters[pos]])
        bench.append(pos_players.iloc[n_starters[pos]:])
    
    starters_df = pd.concat(starters).reset_index(drop=True)
    bench_df = pd.concat(bench).reset_index(drop=True)
    
    # captain/vice-captain
    starters_df["role"] = "Starter"
    starters_df.loc[0, "role"] = "Captain"
    starters_df.loc[1, "role"] = "Vice-Captain"
    bench_df["role"] = "Bench"
    
    team_df = pd.concat([starters_df, bench_df]).reset_index(drop=True)
    return team_df

position_colors = {
    "GK": "#FFD700",  # Gold
    "DEF": "#00BFFF", # DeepSkyBlue
    "MID": "#32CD32", # LimeGreen
    "FWD": "#FF4500", # OrangeRed
}

if st.button("Generate Team (Table View)"):
    team_df = select_best_team(players_df)
    
    # Highlight captain and vice-captain
    def highlight_roles(row):
        if row.role == "Captain":
            return ['background-color: #FF6347']*len(row)  # Tomato
        elif row.role == "Vice-Captain":
            return ['background-color: #FFA500']*len(row)  # Orange
        elif row.role == "Bench":
            return ['background-color: #D3D3D3']*len(row)  # LightGray
        else:
            return ['']*len(row)

    st.subheader("🏆 Selected Team (Table)")
    st.dataframe(team_df.style.apply(highlight_roles, axis=1))
