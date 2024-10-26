from kfp import dsl
from typing import Dict


@dsl.component(base_image="python:3.10")
def say_hello(output: Dict[str, str]) -> Dict[str, str]:
    divmod_output = dict(quotient="a", remainder="a")
    return divmod_output
