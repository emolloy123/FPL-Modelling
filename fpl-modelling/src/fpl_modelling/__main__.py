"""fpl_modelling file for ensuring the package is executable
as `fpl-modelling` and `python -m fpl_modelling`
"""
import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession


def main(*args, **kwargs) -> Any:
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    interactive = hasattr(sys, 'ps1')
    kwargs["standalone_mode"] = not interactive

    run = find_run_command(package_name)
    return run(*args, **kwargs)

def run_pipeline_programmatically(pipeline_name: str, params: dict = None, env: str= None):


    package_name = Path(__file__).parent.name

    configure_project(package_name)

    with KedroSession(session_id='abc', package_name="package_name").create(env=env, runtime_params=params) as session:
        pipeline_outputs = session.run(
            pipeline_name=pipeline_name
        )
    return pipeline_outputs
if __name__ == "__main__":
    main()
