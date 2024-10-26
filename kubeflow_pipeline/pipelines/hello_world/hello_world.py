import os
from kfp import dsl
from typing import Dict

from kubeflow_pipeline import components


@dsl.pipeline
def pipeline_func() -> Dict[str, str]:
    message = os.environ["message"]
    hello_task = components.hello_world.say_hello(message)
    hello_task.set_display_name("STEP 0: Hello World")
    return hello_task.output
