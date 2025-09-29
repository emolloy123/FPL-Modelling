from .TeamOptimizer import TeamOptimizer
import pandas as pd 
import numpy as np 
import logging

logger = logging.getLogger(__name__)

def pick_optimal_team(players_df: pd.DataFrame, objective_col: str):

    sub_df = players_df[players_df['minutes']>180]

    # Ensure player is not injured or anything
    sub_df = sub_df[sub_df['chance_of_playing_next_round'].isin([np.nan, 100])]

    optimizer = TeamOptimizer(sub_df, objective_col)

    res =  optimizer.solve()

    logger.info(res)

    return res