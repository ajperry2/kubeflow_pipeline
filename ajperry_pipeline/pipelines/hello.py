
from kfp import dsl

from ajperry_pipeline import components


@dsl.pipeline
def hello_world(message: str) -> str:
    hello_task = components.say_hello(message=message)
    hello_task.set_display_name("STEP 0: Hello World")
    return hello_task.output
