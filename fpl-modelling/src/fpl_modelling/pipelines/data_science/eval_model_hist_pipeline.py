"""
Kedro pipeline for evaluating FPL prediction models on historical data.

This pipeline evaluates a model's performance across multiple gameweeks by:
1. Training on all previous gameweeks
2. Predicting the next gameweek
3. Calculating performance metrics
4. Selecting optimal teams based on predictions
"""

from kedro.pipeline import Pipeline, node, pipeline
from .eval_model_hist_nodes import eval_model, compare_pred_team_to_true_score


def create_eval_model_hist_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for historical model evaluation.
    
    The pipeline evaluates a specified model configuration across all available
    gameweeks, producing metrics and optimal team selections for each week.
    
    Args:
        **kwargs: Additional keyword arguments (unused, for Kedro compatibility)
        
    Returns:
        Kedro Pipeline with a single evaluation node
        
    Inputs:
        players_hist_merged: Historical player data with features and targets
        
    Outputs:
        metrics: Dictionary mapping gameweek -> evaluation metrics (MAE, RMSE, R2)
        picked_teams: Dictionary mapping gameweek -> optimal team selections
        
    Parameters:
        model_config: Dict of model configurations ("model_1", "model_2", etc.)
        model_num: Integer selecting which model config to evaluate
        min_gameweek: Minimum gameweek to start evaluation from (default: 2)
        target_col: Name of target column to predict (default: 'next_week_round_points')
    """
    return pipeline([
        node(
            func=eval_model,
            inputs=dict(
                players_hist_merged="players_hist_merged",
                model_config="params:model_config",
                model_num="params:model_num",
                min_gameweek="params:min_gameweek",
                target_col="params:target_col",
            ),
            outputs=["metrics", "picked_teams"],
            name="eval_model_node",
        ),
        node(
            func=compare_pred_team_to_true_score,
            inputs=dict(
                players_hist_merged="players_hist_merged",
                picked_teams="picked_teams",
            ),
            outputs="joined_data",
            name="compare_pred_team_to_true_score",
        ),
    ])