from kfp import dsl


@dsl.component
def say_hello(message: str) -> str:
    print(message)
    return message
