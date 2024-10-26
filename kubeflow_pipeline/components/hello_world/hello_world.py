from typing import Dict

from kfp import dsl


@dsl.component(base_image="python:3.10")
def say_hello(message: str) -> str:
    print(message)
    return message
