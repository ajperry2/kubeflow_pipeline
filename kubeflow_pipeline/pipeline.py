import os

from .utils import KFPClientManager
import kubeflow_pipeline


def launch():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m template_pipeline` and `$ template_pipeline `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    """
    assert "DEPLOYKF_HOST" in os.environ, "Host of deploykf instance required"
    assert "DEPLOYKF_NAMESPACE" in os.environ, "Name of Experiment required"
    assert "DEPLOYKF_USER" in os.environ, "Deploykf username required"
    assert "DEPLOYKF_PASSWORD" in os.environ, "Deploykf password required"
    assert "DEPLOYKF_EXPERIMENT" in os.environ, "Deploykf experiment required"
    assert "DEPLOYKF_RUN" in os.environ, "Deploykf run required"
    assert "PIPELINE_NAME" in os.environ, "Deploykf pipeline name required"

    deploykf_host = os.environ["DEPLOYKF_HOST"]
    deploykf_namespace = os.environ["DEPLOYKF_NAMESPACE"]
    deploykf_username = os.environ.get("DEPLOYKF_USER")
    deploykf_password = os.environ.get("DEPLOYKF_PASSWORD")
    deploykf_experiment = os.environ.get("DEPLOYKF_EXPERIMENT")
    deploykf_run = os.environ.get("DEPLOYKF_RUN")
    pipeline_func = getattr(
        kubeflow_pipeline.pipelines, os.environ.get("PIPELINE_NAME")
    )
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
        arguments={},
        experiment_name=deploykf_experiment,
        run_name=deploykf_run,
        namespace=deploykf_namespace,
    )


if __name__ == "__main__":
    launch()
