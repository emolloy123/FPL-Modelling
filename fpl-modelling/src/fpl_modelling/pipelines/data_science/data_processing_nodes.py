import pandas as pd
import typing as tp 
from .feature_engineering import FeatureEngineeringPipeline
def preprocess_data(df: pd.DataFrame, model_config: tp.Dict, model_num: int):
    """
    Do some intial filters on the data
    """

    # df[f"next_week_round_points"] = df.groupby('player_id')['round_points'].shift(-1)

    # Remove players with no minutes

   
    if df['cumsum_minutes'].max()==0:
        minute_threshold = 0
    else:
         minute_threshold = model_config[model_num]['minute_threshold']
    players_hist_merged_clean = df[df['cumsum_minutes']>minute_threshold]    

    return players_hist_merged_clean


# Convenience function
def eng_rolling_avg_features(
    df: pd.DataFrame,
    rolling_features: tp.Optional[tp.Dict],
    player_col: str = "player_id",
    time_col: str = "round"
) -> pd.DataFrame:
    """Generate rolling average features using configurable methods."""
    pipeline = FeatureEngineeringPipeline(player_col, time_col, rolling_features)
    return pipeline.fit_transform(df)