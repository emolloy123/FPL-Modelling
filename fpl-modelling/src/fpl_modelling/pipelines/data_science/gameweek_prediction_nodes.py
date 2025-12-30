import pandas as pd 
import sklearn 
import typing as tp  
import mlflow 

def points_prediction(df: pd.DataFrame, model_config: tp.Dict, model_num: int, mlflow_tracking_uri: str, predicting_gameweek: int, trained_pipeline=None):
    """
    Predict expected points for all players in the specified gameweek
    """
    predicting_gameweek = predicting_gameweek-1
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if trained_pipeline is None:
        trained_pipeline= mlflow.sklearn.load_model(f"models:/model_gameweek_{predicting_gameweek}/latest")

    features = model_config[model_num]['features']['num_features'] + model_config[model_num]['features']['cat_features']

    
    X = df[df['round']==predicting_gameweek][features]
    y_pred = trained_pipeline.predict(X)
    players_df = df[df['round']==predicting_gameweek].copy()

    # Add or replace the points column with model predictions
    players_df['predicted_next_week_points'] = y_pred
    print(players_df)
    return players_df

