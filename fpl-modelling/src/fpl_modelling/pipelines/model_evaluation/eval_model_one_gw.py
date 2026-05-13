import pandas as pd
import typing as tp
from .ModelEvaluator import ModelEvaluator
import mlflow
import numpy as np
from .plotting import plot_pred_vs_true_per_player, plot_y_true_vs_pred 



def regression_metrics(y_train:np.ndarray, y_test :np.ndarray, y_pred_test :np.ndarray, y_pred_train :np.ndarray, predicting_gameweek: int, mlflow_run_id):

    model_evaluator = ModelEvaluator(predicting_gameweek=predicting_gameweek)

    model_metrics = model_evaluator.get_model_metrics(y_train, y_test, y_pred_train, y_pred_test,  mlflow_run_id= mlflow_run_id)

    fig_test = plot_y_true_vs_pred(y_test, y_pred_test, dataset_label="test")
    mlflow.log_figure(fig_test, "y_true_vs_pred_test.html")

    # Train plot
    fig_train = plot_y_true_vs_pred(y_train, y_pred_train, dataset_label="train")
    mlflow.log_figure(fig_train, "y_true_vs_pred_train.html")

    return model_metrics

def eval_predicted_optimal_team(players_hist_merged: pd.DataFrame, optimiser_res, predicting_gameweek: int, average_points: tp.Dict, mlflow_run_id):

    # 1. Get corrected points for any change to captain and/or subsitutions and account for double gameweeks

    # a. account for players with two matches in same gameweek
    players_hist_merged_by_gw = (
            players_hist_merged[
                players_hist_merged['round'] == predicting_gameweek
            ]
            .groupby('player_id', as_index=False)
            .agg({
                'player_name': 'first',
                'player_id': 'first',
                'next_week_round_points': 'sum',
                'next_week_round_minutes': 'sum',
                'position_name': 'first'
            })
        )
    
    # b. Join predictions to known data
    predictions = optimiser_res['squad_ranking'][['player_name', 'predicted_next_round_points', 'rank']]

    data_w_predictions_by_gw = predictions.merge(
            players_hist_merged_by_gw, 
            on='player_name', 
            how='inner'
        ).sort_values(by='rank')

    # c. Make subsitutions appropriately
    gw_starters = make_subsitions(data_w_predictions_by_gw)

    gw_starters_data = data_w_predictions_by_gw[data_w_predictions_by_gw['player_name'].isin(gw_starters['player_name'])]

    print(gw_starters_data[['player_name', 'predicted_next_round_points', 'next_week_round_points']])

    average_points_that_gw = average_points[predicting_gameweek]

    # mlflow.start_run(mlflow_run_id)
    fig = plot_pred_vs_true_per_player(gw_starters_data)
    mlflow.log_figure(fig, "pred_vs_true_per_player.html")

    mlflow.log_metrics({
        'predicted_team_total_true_points': gw_starters_data['next_week_round_points'].sum(),
        'predicted_team_total_predicted_points': gw_starters_data['predicted_next_round_points'].sum(),
        'average_points_that_gameweek': average_points_that_gw
        })

    return gw_starters_data

def make_subsitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FPL autosubs:
    - Replace starters with 0 minutes using bench players
    - Respect formation constraints
    - Bench priority based on rank (lower rank = higher priority)
    """

    df = df.copy()

    # Split starters and bench
    starters = df[df['rank'] <= 11].copy()
    bench = df[df['rank'] > 11].copy().sort_values('rank')  # priority order

    # Formation constraints
    MIN_FORMATION = {
        'Goalkeeper': 1,
        'Defender': 3,
        'Midfielder': 2,
        'Forward': 1
    }

    MAX_FORMATION = {
        'Goalkeeper': 1,
        'Defender': 5,
        'Midfielder': 5,
        'Forward': 3
    }

    def is_valid_formation(players: pd.DataFrame) -> bool:
        counts = players['position_name'].value_counts().to_dict()
        for pos in MIN_FORMATION:
            if counts.get(pos, 0) < MIN_FORMATION[pos]:
                return False
            if counts.get(pos, 0) > MAX_FORMATION[pos]:
                return False
        return True

    # --- STEP 1: Handle GK separately ---
    gk = starters[starters['position_name'] == 'Goalkeeper']

    if len(gk) == 1 and gk.iloc[0]['next_week_round_minutes'] == 0:
        bench_gk = bench[bench['position_name'] == 'Goalkeeper']
        if not bench_gk.empty:
            sub = bench_gk.iloc[0]

            # swap
            starters = starters[starters['player_id'] != gk.iloc[0]['player_id']]
            starters = pd.concat([starters, sub.to_frame().T])

            bench = bench[bench['player_id'] != sub['player_id']]

    # --- STEP 2: Handle outfield players ---
    zero_min_starters = starters[
        (starters['next_week_round_minutes'] == 0) &
        (starters['position_name'] != 'Goalkeeper')
    ].copy()

    for _, starter in zero_min_starters.iterrows():

        replaced = False

        for _, sub in bench.iterrows():

            temp_starters = starters[
                starters['player_id'] != starter['player_id']
            ]
            temp_starters = pd.concat([temp_starters, sub.to_frame().T])

            if is_valid_formation(temp_starters):
                # perform substitution
                starters = temp_starters
                bench = bench[bench['player_id'] != sub['player_id']]
                replaced = True
                break

        # if no valid sub found → player stays (FPL behaviour)

    return starters
