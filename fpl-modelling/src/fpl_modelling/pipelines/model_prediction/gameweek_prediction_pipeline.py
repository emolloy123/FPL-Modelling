from kedro.pipeline import Pipeline, node, pipeline

from .gameweek_prediction_nodes import model_prediction_train_test, get_predicted_optimal_team_next_gameweek, join_back_predictions

def create_gameweek_prediction_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for model training using a flexible model config dict.
    """
    return pipeline([
        node(
            func=model_prediction_train_test,
            inputs=dict(
                pipeline = "trained_pipeline",
                X_test = "X_test",
                X_train = "X_train"
            ),  
            outputs=["y_pred_train", "y_pred_test"],
            name="model_prediction_train_test_node",
        ),
        node(
            func=join_back_predictions,
            inputs=dict(
                df_test = "df_test",
                df_train = "df_train",
                y_pred_test = "y_pred_test",
                y_pred_train = "y_pred_train"
            ),  
            outputs=["df_train_w_pred", "df_test_w_pred"],
            name="join_back_predictions_node",
        ),
        
        node(
            func=get_predicted_optimal_team_next_gameweek,
            inputs=dict(
                df_test = "df_test_w_pred"
            ),  
            outputs="predicted_optimal_team_next_gw",
            name="get_predicted_optimal_team_next_gameweek_node",
        ),
    ])
    
