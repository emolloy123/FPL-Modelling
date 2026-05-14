import typing as tp


import importlib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModelPipelineBuilder:
    """
    Dynamically builds a scikit-learn pipeline (preprocessor + model)
    from a configuration dictionary.

    Example usage:
        builder = ModelPipelineBuilder(model_config)
        pipeline = builder.build_pipeline()
    """

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def _import_from_string(class_path: str):
        """Dynamically import a class from its full path string."""
        module_name, class_name = class_path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)

    def _build_preprocessor(self):
        """Construct a ColumnTransformer from config."""
        preprocessor_cfg = self.config.get("preprocessor", {})
        steps_cfg = preprocessor_cfg.get("steps", [])

        if not steps_cfg:
            return None

        transformers = []

        for step in steps_cfg:
            transformer_class = self._import_from_string(step["transformer"])

            params = step.get("params", {})

            transformer = transformer_class(**params)

            transformers.append(
                (step["name"], transformer, step["columns"])
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

    def _build_model(self):
        """Instantiate the model from config."""
        model_info = self.config["model"]
        model_class = self._import_from_string(model_info["class"])
        hyperparams = model_info.get("hyperparams", {})
        return model_class(**hyperparams)

    def build_pipeline(self):
        """Assemble the full sklearn Pipeline."""
        preprocessor = self._build_preprocessor()
        model = self._build_model()

        steps = []
        if preprocessor is not None:
            steps.append(("preprocessor", preprocessor))
        steps.append(("model", model))

        pipeline = Pipeline(steps=steps)
        return pipeline

def load_model_config(model_config: tp.Dict, model_num: int): 
    """
    Load model pipeline
    """

    model_config = model_config[model_num]
    builder = ModelPipelineBuilder(model_config) 
    pipeline = builder.build_pipeline()

    features = model_config['features']['num_features'] + model_config['features']['cat_features']

    return pipeline, features