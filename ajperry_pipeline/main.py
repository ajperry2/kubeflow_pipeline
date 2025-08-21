import typer
import random
import string
from typing import Optional
from typing_extensions import Annotated
import json
import kfp

from .utils import get_istio_auth_session
import ajperry_pipeline

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
        ajperry_pipeline.pipelines, pipeline
    )
    
    auth_session = get_istio_auth_session(
        url=f"https://{host}",
        username=username,
        password=password
    )
    client = kfp.Client(
        host=f"https://{host}/pipeline",
        cookies=auth_session["session_cookie"]
    )
    # Get definition of experiment/run
    # Make experiment if it does not exist
    try:
        client.get_experiment(
            experiment_name=experiment, namespace=namespace
        )
    except ValueError:
        client.create_experiment(
            name=experiment, namespace=namespace
        )
    client.create_run_from_pipeline_func( 
        pipeline_func=pipeline_func,
        arguments=json.loads(args),
        experiment_name=experiment,
        run_name=run,
        namespace=namespace,
    )
