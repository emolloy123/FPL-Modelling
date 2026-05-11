import typing as tp 
import pandas as pd 
import sklearn 
import mlflow
from fpl_modelling.ModelConfig import load_model_config


def cutoff_future_data(df:pd.DataFrame, predicting_gameweek: int):
    """
    Split data into train and test, test dat being num_test_gameweeks most recent gameweeks
    """

    train_df = df[df['round'] <= predicting_gameweek].reset_index(drop=True)

    return train_df


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, pipeline: sklearn.pipeline.Pipeline, 
                predicting_gameweek: int, mlflow_tracking_uri: str = None):

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(f"gameweek_{predicting_gameweek}")
        
    pipeline.fit(X_train, y_train)

    if mlflow_tracking_uri:
        with mlflow.start_run(run_name="fpl_model_training") as run:

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"model_gameweek_{predicting_gameweek}",
            )

    return pipeline

# def eval_model_walk_forward_cross_val():
