import pandas as pd
import typing as tp 
from .feature_engineering import FeatureEngineeringPipeline
from fpl_modelling.ModelConfig import load_model_config

def filter_players_training_data(players_hist_merged: pd.DataFrame, model_config: tp.Dict, model_num: int):

    minute_threshold = model_config[model_num]['minute_threshold']

    # if its just predicting the second gameweek minute therhosld should be 0
    if players_hist_merged['cumsum_minutes'].max()==0:
        minute_threshold = 0

    return players_hist_merged[players_hist_merged['cumsum_minutes']>minute_threshold]    

def filter_players_prediction_data(players_hist_merged: pd.DataFrame):

    # remove players not playing in next gameweek
    return players_hist_merged[players_hist_merged['next_week_fixture_count']>0]   

# Convenience function
def eng_rolling_avg_features(
    players_hist_merged: pd.DataFrame,
    rolling_features: tp.Optional[tp.Dict],
    player_col: str = "player_id",
    time_col: str = "round"
) -> pd.DataFrame:
    """Generate rolling average features using configurable methods."""
    pipeline = FeatureEngineeringPipeline(player_col, time_col, rolling_features)
    return pipeline.fit_transform(players_hist_merged)

def train_test_split_by_gw(model_num: int, df: pd.DataFrame, model_config: tp.Dict, predicting_gameweek: int):
    
    pipeline, features = load_model_config(model_config, model_num)

    X_test = df[df['round']==predicting_gameweek][features]

    X_train = df[df['round']<predicting_gameweek][features]

    y_test = df[df['round']==predicting_gameweek]['next_week_round_points']

    y_train = df[df['round']<predicting_gameweek]['next_week_round_points']

    return X_train, y_train, X_test, y_test