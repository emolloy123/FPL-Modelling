def flatten_model_config_for_mlflow(model_config: dict, model_num: int) -> dict:
    """Flatten nested model config into MLflow-compatible flat params dict."""
    
    cfg = model_config[model_num]
    params = {}

    # Top level
    params["model_num"] = model_num
    params["minute_threshold"] = cfg["minute_threshold"]

    # Features
    params["num_features"] = ", ".join(cfg["features"]["num_features"])
    params["cat_features"] = ", ".join(cfg["features"]["cat_features"])
    params["n_num_features"] = len(cfg["features"]["num_features"])
    params["n_cat_features"] = len(cfg["features"]["cat_features"])

    # Preprocessor steps
    for step in cfg["preprocessor"]["steps"]:
        params[f"preprocessor_{step['name']}_transformer"] = step["transformer"]
        params[f"preprocessor_{step['name']}_columns"] = ", ".join(step["columns"])
        for k, v in step.get("params", {}).items():
            params[f"preprocessor_{step['name']}_{k}"] = v

    # Model
    params["model_class"] = cfg["model"]["class"]
    for k, v in cfg["model"]["hyperparams"].items():
        params[f"hyperparam_{k}"] = v

    return params