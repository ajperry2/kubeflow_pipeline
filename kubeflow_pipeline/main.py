import os
import typer
import random
import string
from typing import Optional
from typing_extensions import Annotated
import json

from .utils import KFPClientManager
import kubeflow_pipeline

app = typer.Typer()

@app.command()
def launch_run(
    host: Annotated[str, typer.Option()],
    namespace: Annotated[str, typer.Option()],
    username: Annotated[str, typer.Option()],
    password: Annotated[str, typer.Option()],
    pipeline: Annotated[str, typer.Option()],
    experiment: Annotated[str, typer.Option()],
    run: Annotated[Optional[str], typer.Option()] = None,
    args: Annotated[Optional[str], typer.Option()] = "{}",
):  

    if run is None:
        alphabet = string.ascii_letters + string.digits
        run = ''.join(random.choice(alphabet) for i in range(20))

    pipeline_func = getattr(
        kubeflow_pipeline.pipelines, pipeline
    )
    

    kfp_client_manager = KFPClientManager(
        api_url="https://" + host + "/pipeline",
        skip_tls_verify=False,
        dex_username=username,
        dex_password=password,
        dex_auth_type="local",
    )
    kfp_client = kfp_client_manager.create_kfp_client()
    # Get definition of experiment/run
    # Make experiment if it does not exist
    try:
        kfp_client.get_experiment(
            experiment_name=experiment, namespace=namespace
        )
    except ValueError:
        kfp_client.create_experiment(
            name=experiment, namespace=namespace
        )
    kfp_client.create_run_from_pipeline_func( 
        pipeline_func=pipeline_func,
        arguments=json.loads(args),
        experiment_name=experiment,
        run_name=run,
        namespace=namespace,
    )
