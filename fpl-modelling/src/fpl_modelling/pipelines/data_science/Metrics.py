from collections import defaultdict 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
import pandas as pd 

class Metrics:
    def __init__(self):
        self.gameweek_metrics = defaultdict(
        lambda: {'train': {}, 'test': {}}
    ) 
    
    @staticmethod
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
    
    def calculate_metrics_at_gameweek(self, gameweek, y_pred_test, y_true_test, y_pred_train, y_true_train):

        self.gameweek_metrics[gameweek]['test'] = Metrics.compute_regression_metrics(
        y_true_test, y_pred_test
        )

        self.gameweek_metrics[gameweek]['train'] = Metrics.compute_regression_metrics(
            y_true_train, y_pred_train
        )
    
    def get_metrics_df(self):
        rows = []

        for gw, splits in self.gameweek_metrics.items():
            row = {'gameweek': gw}

            for metric in splits['train'].keys():
                row[f'{metric}_train'] = splits['train'][metric]
                row[f'{metric}_test'] = splits['test'][metric]

            rows.append(row)

        metrics_df = pd.DataFrame(rows).sort_values('gameweek').reset_index(drop=True)

        return metrics_df