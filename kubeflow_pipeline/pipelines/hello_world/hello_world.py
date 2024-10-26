import os

from kfp import dsl

from kubeflow_pipeline import components


@dsl.pipeline
def hello_world() -> str:
    message = os.environ.get("MESSAGE", "")
    hello_task = components.hello_world.say_hello(message=message)
    hello_task.set_display_name("STEP 0: Hello World")
    return hello_task.output
