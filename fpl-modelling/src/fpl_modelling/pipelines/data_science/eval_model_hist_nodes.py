"""
Nodes for evaluating FPL prediction models across historical gameweeks.

This module provides functions to:
- Generate predictions on train/test splits
- Calculate evaluation metrics per gameweek
- Pick optimal fantasy football teams based on predictions
- Evaluate models across multiple gameweeks in a time-series manner
"""

import pandas as pd
import typing as tp
import logging
from collections import defaultdict

import numpy as np
import sklearn
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score
)

from .train_model_nodes import load_config, train_test_split, train_model
from fpl_modelling.pipelines.optimisation.pick_team_nodes import pick_optimal_team
from .Metrics import Metrics

logger = logging.getLogger(__name__)

def _get_train_test_predictions(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    features: tp.List[str],
    model: sklearn.pipeline.Pipeline,
    target_col: str
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and extract ground truth for train and test sets.
    
    Private helper function - not a Kedro node.
    
    Args:
        test_df: Test dataset with features and target column
        train_df: Training dataset with features and target column
        features: List of feature column names to use for prediction
        model: Trained sklearn pipeline/model
        target_col: Name of the target column to predict
        
    Returns:
        Tuple containing:
            - y_pred_test: Model predictions on test set
            - y_true_test: Actual values for test set
            - y_pred_train: Model predictions on train set
            - y_true_train: Actual values for train set
            
    Raises:
        ValueError: If required columns are missing from dataframes
        KeyError: If specified features are not in dataframes
    """
    # Validate required columns exist
    for df, df_name in [(test_df, 'test_df'), (train_df, 'train_df')]:
        if target_col not in df.columns:
            raise ValueError(
                f"{df_name} missing required target column: '{target_col}'"
            )
        
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise KeyError(
                f"{df_name} missing required features: {missing_features}"
            )
    
    try:
        # Generate predictions
        y_pred_test = model.predict(test_df[features])
        y_pred_train = model.predict(train_df[features])
        
        # Extract ground truth
        y_true_test = test_df[target_col].values
        y_true_train = train_df[target_col].values
        
        logger.info(
            f"Generated predictions - Test samples: {len(y_pred_test)}, "
            f"Train samples: {len(y_pred_train)}"
        )
        
        return y_pred_test, y_true_test, y_pred_train, y_true_train
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise


def _get_optimal_team_test_prediction(
    df: pd.DataFrame,
    y_pred_test: np.ndarray,
    y_true_test: np.ndarray,
    objective_col_name: str
) -> tp.Dict:
    """
    Pick optimal fantasy football team based on model predictions.
    
    Private helper function - not a Kedro node.
    
    Creates a copy of the input dataframe with prediction columns added,
    then runs optimization to select the best team.
    
    Args:
        df: Player data for the gameweek
        y_pred_test: Model predictions for next week points
        y_true_test: Actual next week points (for comparison)
        
    Returns:
        Dictionary containing the optimal team selection with predicted scores
        
    Raises:
        ValueError: If prediction arrays don't match dataframe length
    """
    if len(y_pred_test) != len(df):
        raise ValueError(
            f"Prediction length ({len(y_pred_test)}) doesn't match "
            f"dataframe length ({len(df)})"
        )
    
    if len(y_true_test) != len(df):
        raise ValueError(
            f"Ground truth length ({len(y_true_test)}) doesn't match "
            f"dataframe length ({len(df)})"
        )
    
    try:
        # Create a copy to avoid modifying the original dataframe
        df_with_predictions = df.copy()
        df_with_predictions[objective_col_name] = y_pred_test
        df_with_predictions['true_next_week_points'] = y_true_test
        
        logger.debug(
            f"Picking optimal team from {len(df_with_predictions)} players"
        )
        
        # Run optimization without printing (keep output clean for Kedro logs)
        opt_team = pick_optimal_team(df_with_predictions, objective_col=objective_col_name, print_sol=False)
        
        logger.info(f"Optimal team selected successfully")
        
        return opt_team
        
    except Exception as e:
        logger.error(f"Error picking optimal team: {str(e)}")
        raise


def eval_model(
    players_hist_merged: pd.DataFrame,
    model_config: tp.Dict,
    model_num: int,
    min_gameweek: int = 2,
    target_col: str = 'next_week_round_points'
) -> tp.Tuple[tp.Dict, tp.Dict]:
    """
    Evaluate a model across multiple gameweeks using time-series cross-validation.
    
    Main Kedro node function.
    
    For each gameweek from min_gameweek to the last available gameweek:
    1. Split data into train (all prior gameweeks) and test (current gameweek)
    2. Train model on training data
    3. Generate predictions and calculate metrics
    4. Pick optimal team based on predictions
    
    Args:
        players_hist_merged: Historical player data with features and targets.
            Required columns: 'round' and the target column specified by target_col,
            plus all features specified in model_config
        model_config: Configuration dictionary containing model specifications,
            preprocessing steps, and features for different model variants
        model_num: Index of the model configuration to use from model_config
        min_gameweek: Minimum gameweek to start evaluation from. Defaults to 2
            (requires at least one previous gameweek for training)
        target_col: Name of the target column to predict. Defaults to 
            'next_week_round_points'
        
    Returns:
        Tuple containing:
            - gameweek_metrics: Dict mapping gameweek -> evaluation metrics
              (MAE, RMSE, R2, etc. for both train and test sets)
            - picked_teams: Dict mapping gameweek -> optimal team selection
              with predicted and actual point values
              
    Raises:
        ValueError: If dataframe is empty, missing required columns,
                   or has insufficient gameweeks
        KeyError: If model_num doesn't exist in model_config
        
    """
    # Validate input dataframe
    if players_hist_merged.empty:
        raise ValueError("Input dataframe 'players_hist_merged' is empty")
    
    # Check for structural columns and target
    required_cols = ['round'] + [target_col]
    for col in required_cols:
        if col not in players_hist_merged.columns:
            raise ValueError(
                f"Input dataframe missing required column: '{col}'"
            )
    
    max_gameweek = players_hist_merged['round'].max()
    min_gameweek_data = players_hist_merged['round'].min()
    
    if pd.isna(max_gameweek) or pd.isna(min_gameweek_data):
        raise ValueError("'round' column contains NaN values")
    
    if max_gameweek < min_gameweek + 1:
        raise ValueError(
            f"Insufficient gameweeks for evaluation. "
            f"Need at least {min_gameweek + 1}, found {max_gameweek}"
        )
    
    logger.info(
        f"Starting model evaluation from gameweek {min_gameweek} "
        f"to {max_gameweek} (model_num={model_num}, target={target_col})"
    )
    
    # Load model configuration
    try:
        pipeline, features = load_config(model_config, model_num)
        logger.info(f"Loaded model config with {len(features)} features")
    except KeyError:
        raise KeyError(
            f"model_num {model_num} not found in model_config. "
            f"Available models: {list(model_config.keys())}"
        )
    except Exception as e:
        logger.error(f"Error loading model config: {str(e)}")
        raise
    
    # Validate all features exist in dataframe
    missing_features = set(features) - set(players_hist_merged.columns)
    if missing_features:
        raise ValueError(
            f"Model requires features not in dataframe: {missing_features}"
        )
    
    # Initialize containers for results
    metric_handler = Metrics()
    picked_teams = {}
    failed_gameweeks = []
    
    # Evaluate model for each gameweek
    for gameweek in range(min_gameweek, int(max_gameweek)):
        logger.info(f"{'='*20} Gameweek {gameweek} {'='*20}")
        
        try:
            # Create train/test split
            train_df, test_df = train_test_split(
                df=players_hist_merged,
                predicting_gameweek=gameweek
            )
            
            if train_df.empty:
                logger.warning(
                    f"Skipping gameweek {gameweek}: empty training set"
                )
                failed_gameweeks.append(gameweek)
                continue
                
            if test_df.empty:
                logger.warning(
                    f"Skipping gameweek {gameweek}: empty test set"
                )
                failed_gameweeks.append(gameweek)
                continue
            
            logger.debug(
                f"Train size: {len(train_df)}, Test size: {len(test_df)}"
            )
            
            # Train model
            model = train_model(
                train_df=train_df,
                pipeline=pipeline,
                features=features,
                predicting_gameweek=gameweek,
                target_col=target_col
            )
            
            # Generate predictions
            y_pred_test, y_true_test, y_pred_train, y_true_train = \
                _get_train_test_predictions(test_df, train_df, features, model, target_col)
            
            # Calculate metrics
            metric_handler.calculate_metrics_at_gameweek(
                gameweek,
                y_pred_test,
                y_true_test,
                y_pred_train,
                y_true_train
            )
            
            # Pick optimal team
            picked_teams[gameweek] = _get_optimal_team_test_prediction(
                test_df,
                y_pred_test,
                y_true_test,
                objective_col_name="predicted_next_week_points"
            )
            
            logger.info(f"Gameweek {gameweek} evaluation completed successfully")
            
        except Exception as e:
            logger.error(
                f"Error evaluating gameweek {gameweek}: {str(e)}",
                exc_info=True
            )
            failed_gameweeks.append(gameweek)
            continue
    
    # Summary logging
    total_gameweeks = int(max_gameweek) - min_gameweek
    successful_gameweeks = total_gameweeks - len(failed_gameweeks)
    
    logger.info(
        f"Evaluation complete: {successful_gameweeks}/{total_gameweeks} "
        f"gameweeks successful"
    )
    
    if failed_gameweeks:
        logger.warning(f"Failed gameweeks: {failed_gameweeks}")
    
    if successful_gameweeks == 0:
        raise RuntimeError(
            "All gameweeks failed evaluation. Check logs for details."
        )

    print(picked_teams)
    
    return metric_handler.gameweek_metrics, picked_teams

def compare_pred_team_to_true_score(picked_teams: tp.Dict, players_hist_merged: pd.DataFrame):
    all_joined_data = []
    
    for gameweek, gw_team in picked_teams.items():
        picked_team_gw = gw_team['squad']  # Assuming this contains player names
        
        # Filter for this gameweek and these players
        true_players_stats = players_hist_merged[
            (players_hist_merged['round'] == gameweek) & 
            (players_hist_merged['player_name'].isin(picked_team_gw))
        ]
        
        # Get predictions - need to understand your data structure here
        # If squad_ranking is in gw_team:
        predictions = gw_team['squad_ranking'][['player_name', 'predicted_next_week_points', 'rank']]
        
        # Merge actual stats with predictions
        joined_data = true_players_stats.merge(
            predictions, 
            on='player_name', 
            how='inner'
        ).sort_values(by='rank')
        
        comparsion = compare_player_points(joined_data)
        
        all_joined_data.append(joined_data)
    print(joined_data)
    return pd.concat(all_joined_data, ignore_index=True)

def compare_player_points(joined_data): 

    pass