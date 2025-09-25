import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="FPL Optimiser", page_icon="⚽", layout="wide")

# --- Title ---
st.title("⚽ FPL Optimiser vs. Most Popular Team")
st.markdown(
    "Compare the points per gameweek of an **optimised team** vs. "
    "the **most popular FPL team**."
)

# --- Example Data (replace with your real results) ---
weeks = list(range(1, 6))
optimised_points = [55, 62, 48, 71, 60]   # your team per GW
popular_points = [50, 58, 52, 65, 59]     # most popular team per GW

df = pd.DataFrame({
    "Gameweek": weeks,
    "Optimised Team": optimised_points,
    "Most Popular Team": popular_points
})

# --- Plotly Line Chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Gameweek"],
    y=df["Optimised Team"],
    mode="lines+markers",
    name="Optimised Team"
))

fig.add_trace(go.Scatter(
    x=df["Gameweek"],
    y=df["Most Popular Team"],
    mode="lines+markers",
    name="Most Popular Team"
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
st.subheader("📈 Summary")
total_opt = df["Optimised Team"].sum()
total_pop = df["Most Popular Team"].sum()

col1, col2 = st.columns(2)
col1.metric("Optimised Team Total", f"{total_opt} pts")
col2.metric("Most Popular Team Total", f"{total_pop} pts")

if total_opt > total_pop:
    st.success(f"🎉 Optimised team is ahead by {total_opt - total_pop} points!")
else:
    st.warning(f"📉 Optimised team is behind by {total_pop - total_opt} points.")
