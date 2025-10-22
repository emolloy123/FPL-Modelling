from fpl_modelling.__main__ import run_pipeline_programmatically

# run_pipeline_programmatically("update_tables")

run_pipeline_programmatically(pipeline_name="gameweek_prediction", params={"current_gameweek": 8, "model_num": 4})