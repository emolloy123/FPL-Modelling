
import pandas as pd
import typing as tp 
from collections import defaultdict
from .train_model_nodes import load_config, train_test_split, train_model

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score

from fpl_modelling.pipelines.optimisation.pick_team_nodes import pick_optimal_team

def eval_model(players_hist_merged: pd.DataFrame, model_config: tp.Dict, model_num: int):

    pipeline, features = load_config(model_config, model_num)

    gameweek_metrics = defaultdict(
        lambda: {'train': {}, 'test': {}}
    )    
    picked_teams = {}
    for gameweek in range(2, players_hist_merged['round'].max()):

        train_df, test_df = train_test_split(df=players_hist_merged, predicting_gameweek=gameweek)

        model = train_model(train_df=train_df, pipeline=pipeline, features=features, 
                predicting_gameweek=gameweek, target_col='next_week_round_points')
        
        # predictions
        y_pred_test = model.predict(test_df[features])
        y_pred_train = model.predict(train_df[features])

        # ground truth
        y_true_test = test_df['next_week_round_points']
        y_true_train = train_df['next_week_round_points']

        df = test_df.copy()
        players_df = df[df['round']==gameweek].copy()

        # Add or replace the points column with model predictions
        players_df['predicted_next_week_points'] = y_pred_test
        players_df['true_next_week_points'] = y_true_test

        opt_team = pick_optimal_team(players_df, print_sol=False)
        picked_teams[gameweek] = opt_team
        
        opt_team_starters = opt_team['starters']

        chosen_team_true_points = players_df[players_df['player_name'].isin(opt_team_starters)]

        print(chosen_team_true_points[['player_name', 'true_next_week_points']])

        # store metrics
        gameweek_metrics[gameweek]['test'] = compute_regression_metrics(
            y_true_test, y_pred_test
        )

        gameweek_metrics[gameweek]['train'] = compute_regression_metrics(
            y_true_train, y_pred_train
        )
    rows = []

    for gw, splits in gameweek_metrics.items():
        row = {'gameweek': gw}

        for metric in splits['train'].keys():
            row[f'{metric}_train'] = splits['train'][metric]
            row[f'{metric}_test'] = splits['test'][metric]

        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values('gameweek').reset_index(drop=True)

    print(metrics_df)
    print(picked_teams)
    return gameweek_metrics

def compute_regression_metrics(y_true, y_pred):
    """
    Compute standard regression metrics.
    Returns a dict.
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
    }