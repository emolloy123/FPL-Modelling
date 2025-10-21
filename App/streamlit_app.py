import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="FPL Optimiser", page_icon="⚽", layout="wide")

# --- Title ---
st.title("⚽ FPL Optimiser vs. Most Selected Team vs. Average")
st.markdown(
    "Compare the points per gameweek of **My Team**, "
    "**Most Selected Team**, and the **Overall Average**."
)

# --- Example Data (replace with your real results) ---
my_team = [45, 63, 44, 44, 39, 71, 64]
my_team_selection = ['random', 'random', 'random', 'random', 'random', 'optimal wild card']
most_selected_team = [34, 61, 58, 76, 55, 45, 64]
average_points = [54, 51, 48, 63, 42, 46, 60]

weeks = list(range(1, len(my_team)+1))

df = pd.DataFrame({
    "Gameweek": weeks,
    "My Team": my_team,
    "Most Selected Team": most_selected_team,
    "Average Points": average_points
})

# --- Plotly Line Chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Gameweek"],
    y=df["My Team"],
    mode="lines+markers",
    name="My Team"
))

fig.add_trace(go.Scatter(
    x=df["Gameweek"],
    y=df["Most Selected Team"],
    mode="lines+markers",
    name="Most Selected Team"
))

fig.add_trace(go.Scatter(
    x=df["Gameweek"],
    y=df["Average Points"],
    mode="lines+markers",
    name="Average Points"
))

fig.update_layout(
    title="Points per Gameweek",
    xaxis_title="Gameweek",
    yaxis_title="Points",
    template="plotly_white",
    legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")
)

st.plotly_chart(fig, use_container_width=True)

# --- Summary Stats ---
st.subheader("📈 Leaderboard (Total Points)")

totals = {
    "My Team": df["My Team"].sum(),
    "Most Selected Team": df["Most Selected Team"].sum(),
    "Average Points": df["Average Points"].sum()
}

leaderboard = (
    pd.DataFrame(totals.items(), columns=["Team", "Total Points"])
    .sort_values(by="Total Points", ascending=False)
    .reset_index(drop=True)
)

# Offset index so it starts at 1
leaderboard.index = leaderboard.index + 1

st.dataframe(leaderboard, use_container_width=True)

# --- Highlight Winner ---
winner = leaderboard.iloc[0]
st.success(f"🏆 {winner['Team']} is leading with {winner['Total Points']} points!")

