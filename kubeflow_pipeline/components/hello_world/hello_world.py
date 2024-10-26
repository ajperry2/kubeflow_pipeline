from kfp import dsl
from typing import Dict


@dsl.component(base_image="python:3.10")
def say_hello(message: str) -> str:
    print(message)
    return message
