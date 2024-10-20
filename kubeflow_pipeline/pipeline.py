import os

from kfp import dsl

from .utils import KFPClientManager


@dsl.component
def say_hello(name: str) -> str:
    hello_text = f"Hello, {name}!"
    print(hello_text)
    return hello_text


@dsl.pipeline
def pipeline_func(recipient: str) -> str:
    return say_hello(name=recipient).output


def launch():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m template_pipeline` and `$ template_pipeline `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    """
    assert "DEPLOYKF_HOST" in os.environ, "Host of deploykf instance required"
    assert "DEPLOYKF_NS" in os.environ, "Deploykf namespace required"
    deploykf_host = os.environ["deploykf_host"]
    deploykf_namespace = os.environ["INPUT_DEPLOYKF_NAMESPACE"]
    deploykf_username = os.environ.get("INPUT_DEPLOYKF_USERNAME", "")
    deploykf_password = os.environ.get("INPUT_DEPLOYKF_PASSWORD", "")

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
    assert "INPUT_EXPERIMENT" in os.environ, "Name of Experiment required"
    assert "INPUT_RUN" in os.environ, "Name of run required"
    experiment_name = os.environ["INPUT_EXPERIMENT"]
    run_name = os.environ["INPUT_RUN"]
    # Make experiment if it does not exist
    try:
        kfp_client.get_experiment(
            experiment_name=experiment_name, namespace=deploykf_namespace
        )
    except ValueError:
        kfp_client.create_experiment(
            name=experiment_name, namespace=deploykf_namespace
        )
    kfp_client.create_run_from_pipeline_func(
        pipeline_func=pipeline_func,
        arguments={"recipient": "github"},
        experiment_name=experiment_name,
        run_name=run_name,
        namespace=deploykf_namespace,
    )
