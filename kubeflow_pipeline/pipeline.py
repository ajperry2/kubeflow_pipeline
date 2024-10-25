import os

from kfp import dsl
from typing import Dict

from .utils import KFPClientManager
from kubeflow_pipeline import components


@dsl.pipeline
def pipeline_func(component_names: str) -> Dict[str, str]:
    current_output: Dict[str, str] = dict()
    for i, component_name in enumerate(component_names.split(",")):
        task_obj = getattr(components, component_name)
        task = task_obj(**current_output)
        task.set_display_name(f"STEP {i}: {component_name}")
        task.set_caching_options(False)
        current_output = task.output
    return current_output


def launch():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m template_pipeline` and `$ template_pipeline `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    """
    assert "deploykf_host" in os.environ, "Host of deploykf instance required"
    assert "deploykf_namespace" in os.environ, "Name of Experiment required"
    assert "deploykf_user" in os.environ, "Deploykf username required"
    assert "deploykf_password" in os.environ, "Deploykf password required"
    assert "deploykf_experiment" in os.environ, "Deploykf experiment required"
    assert "deploykf_run" in os.environ, "Deploykf run required"
    deploykf_host = os.environ["deploykf_host"]
    deploykf_namespace = os.environ["deploykf_namespace"]
    deploykf_username = os.environ.get("deploykf_user")
    deploykf_password = os.environ.get("deploykf_password")
    deploykf_experiment = os.environ.get("deploykf_experiment")
    deploykf_run = os.environ.get("deploykf_run")
    component_names = os.environ.get("component_names")
    # initialize a credentials instance and client

    # Security Note: As all deployments are routed through my routers iptable,
    # I am not too concerned about MITM attacks (so lack of ssl encryption is
    # fine for now). If others are connecting over the internet, be sure to
    # Setup https and set "skip_tls_verify" to False
    kfp_client_manager = KFPClientManager(
        api_url="https://" + deploykf_host + "/pipeline",
        skip_tls_verify=False,
        dex_username=deploykf_username,
        dex_password=deploykf_password,
        dex_auth_type="local",
    )
    kfp_client = kfp_client_manager.create_kfp_client()
    # Get definition of experiment/run
    # Make experiment if it does not exist
    try:
        kfp_client.get_experiment(
            experiment_name=deploykf_experiment, namespace=deploykf_namespace
        )
    except ValueError:
        kfp_client.create_experiment(
            name=deploykf_experiment, namespace=deploykf_namespace
        )
    kfp_client.create_run_from_pipeline_func(
        pipeline_func=pipeline_func,
        arguments={"component_names": component_names},
        experiment_name=deploykf_experiment,
        run_name=deploykf_run,
        namespace=deploykf_namespace,
    )


if __name__ == "__main__":
    launch()
