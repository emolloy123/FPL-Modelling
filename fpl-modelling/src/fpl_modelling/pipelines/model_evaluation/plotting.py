import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score
import mlflow 
import pandas as pd 
import typing as tp 

def plot_y_true_vs_pred(y_true, y_pred, dataset_label):
    """
    Scatter plot of y_true vs y_pred with y=x reference line
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Line for perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name=f"{dataset_label} data",
        opacity=0.7
    ))

    # y = x line (perfect prediction)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect prediction (R² = 1)",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title=f"True vs Predicted ({dataset_label}) R² = {r2_score(y_true, y_pred):.3f}",
        xaxis_title="True Values",
        yaxis_title="Predicted Values",
        height=500
    )

    return fig

def plot_pred_vs_true_per_player(df):
    """
    Expects df with:
    - player_name
    - predicted_next_round_points
    - next_week_round_points
    """

    df = df.sort_values("predicted_next_round_points", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["player_name"],
        y=df["predicted_next_round_points"],
        name="Predicted"
    ))

    fig.add_trace(go.Bar(
        x=df["player_name"],
        y=df["next_week_round_points"],
        name="True"
    ))

    fig.update_layout(
        title="Predicted vs True Points per Player",
        xaxis_title="Player",
        yaxis_title="Points",
        barmode="group",
        xaxis_tickangle=-45,
        height=500
    )

    return fig

def plot_multi_metrics(
    metrics_df: pd.DataFrame,
    cols_to_plot: tp.List[str],
    mlflow_run_id: str = None,
    title: str = "Metrics by Gameweek"
):
    """
    Plot multiple metrics on the same figure and log to MLflow.

    metrics_df: DataFrame indexed by gameweek
    cols_to_plot: list of metric column names
    """

    fig = go.Figure()

    # Add each metric as a trace
    for col in cols_to_plot:
        fig.add_trace(
            go.Scatter(
                x=metrics_df.index,
                y=metrics_df[col],
                mode="lines+markers",
                name=col.upper()
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Gameweek",
        yaxis_title="Metric Value",
        template="plotly_white",
        legend_title="Metrics"
    )

    # Log to MLflow
    if mlflow_run_id is not None:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_figure(fig, f"{title}.html")

    return fig
    
def plot_metrics_by_gw(metrics_by_gw: pd.DataFrame, mlflow_run_id=None):
    """
    metrics_by_gw: DataFrame indexed by gameweek
                 columns like ['r2', 'mae', 'rmse', 'mape']
    """

    figs = {}

    with mlflow.start_run(run_id=mlflow_run_id):
        for col in metrics_by_gw.columns:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=metrics_by_gw.index,
                    y=metrics_by_gw[col],
                    mode="lines+markers",
                    name=col
                )
            )

            fig.update_layout(
                title=f"{col.upper()} by Gameweek",
                xaxis_title="Gameweek",
                yaxis_title=col.upper(),
                template="plotly_white"
            )

            figs[col] = fig

            # Log each figure separately
            mlflow.log_figure(fig, f"{col}.html")

    return figs