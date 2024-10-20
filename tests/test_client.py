import os

from kubeflow_pipeline.base import NAME


def test_client():
    deploykf_host = os.environ["deploykf_host"]
    deploykf_namespace = os.environ["deploykf_namespace"]
    deploykf_username = os.environ.get("deploykf_user")
    deploykf_password = os.environ.get("deploykf_password")
    deploykf_experiment = os.environ.get("deploykf_experiment")
    deploykf_run = os.environ.get("deploykf_run")
