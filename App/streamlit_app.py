import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="FPL ML Optimiser", layout="wide")

# -----------------------------
# Mock Data Generators
# -----------------------------
np.random.seed(42)
players = [f"Player {i}" for i in range(1, 101)]
positions = np.random.choice(["GK", "DEF", "MID", "FWD"], size=100)
teams = np.random.choice([f"Team {i}" for i in range(1, 21)], size=100)
prices = np.round(np.random.uniform(4.0, 12.5, size=100), 1)

mock_df = pd.DataFrame({
    "player": players,
    "position": positions,
    "team": teams,
    "price": prices,
    "predicted_points": np.random.uniform(2, 10, size=100),
    "actual_points": np.random.uniform(0, 15, size=100)
})

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Model Performance",
    "Team Optimiser",
    "Model Info"
])

# -----------------------------
# Page 1: Model Performance
# -----------------------------
if page == "Model Performance":
    st.title("Model vs Baseline Performance")

    df = mock_df.copy()
    df["baseline_points"] = df["actual_points"].mean()

    st.subheader("Summary Metrics")
    col1, col2 = st.columns(2)

    model_mae = np.mean(np.abs(df["predicted_points"] - df["actual_points"]))
    baseline_mae = np.mean(np.abs(df["baseline_points"] - df["actual_points"]))

    col1.metric("Model MAE", f"{model_mae:.2f}")
    col2.metric("Baseline MAE", f"{baseline_mae:.2f}")

    st.subheader("Predicted vs Actual Points")
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x="predicted_points",
        y="actual_points",
        tooltip=["player", "team"]
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Page 2: Team Optimiser
# -----------------------------
elif page == "Team Optimiser":
    st.title("Optimal Team Selector")

    budget = st.slider("Budget", 80.0, 120.0, 100.0)

    df = mock_df.copy()

    # Simple greedy optimiser (mock)
    df = df.sort_values("predicted_points", ascending=False)

    team = []
    total_cost = 0

    constraints = {
        "GK": 1,
        "DEF": 4,
        "MID": 4,
        "FWD": 2
    }

    counts = {k: 0 for k in constraints}

    for _, row in df.iterrows():
        pos = row["position"]
        if counts[pos] < constraints[pos] and total_cost + row["price"] <= budget:
            team.append(row)
            counts[pos] += 1
            total_cost += row["price"]

    team_df = pd.DataFrame(team)

    st.subheader("Selected Team")
    st.dataframe(team_df)

    st.metric("Total Cost", f"£{total_cost:.1f}")
    st.metric("Expected Points", f"{team_df['predicted_points'].sum():.2f}")

# -----------------------------
# Page 3: Model Info
# -----------------------------
elif page == "Model Info":
    st.title("Model Information")

    st.markdown("""
    ### Overview
    This model predicts Fantasy Premier League (FPL) points for the next gameweek.

    ### Features Used
    - Historical points
    - Form (last 5 games)
    - Minutes played
    - Opponent strength
    - Home/Away

    ### Model Type
    Example: Gradient Boosting Regressor

    ### Evaluation
    - Mean Absolute Error (MAE)
    - Compared against baseline (average points)

    ### Future Improvements
    - Incorporate expected goals (xG)
    - Injury/news sentiment analysis
    - Bayesian uncertainty estimates
    """)

    st.subheader("Sample Data")
    st.dataframe(mock_df.head())
