from fpl_modelling.pipelines.data_processing.ExpandingDF import ExpandingDF
from .ModelPipelineBuilder import ModelPipelineBuilder
import typing as tp 
import pandas as pd 
import sklearn 
import mlflow
import mlflow.sklearn

def load_config(model_config: tp.Dict, model_num: int):
    """
    Load model pipeline
    """

    model_config = model_config[model_num]
    builder = ModelPipelineBuilder(model_config) 
    pipeline = builder.build_pipeline()

    features = model_config['features']['num_features'] + model_config['features']['cat_features']

    return pipeline, features

def train_test_split(expanded_df:pd.DataFrame, num_test_gameweeks: int):
    """
    Split data into train and test, test dat being num_test_gameweeks most recent gameweeks
    """

    train_df, test_df = ExpandingDF.gw_train_test_split(expanded_df, num_test_gameweeks=num_test_gameweeks)

    return train_df, test_df

def train_model(train_df: pd.DataFrame, pipeline: sklearn.pipeline.Pipeline, features: tp.List[str], target_col: str, 
                mlflow_tracking_uri: str, current_gameweek: int):

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"gameweek_{current_gameweek}")
    X = train_df[features]
    y = train_df[target_col]

    with mlflow.start_run(run_name="fpl_model_training") as run:
        pipeline.fit(X, y)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=f"model_gameweek_{current_gameweek}",
        )


    return pipeline
    