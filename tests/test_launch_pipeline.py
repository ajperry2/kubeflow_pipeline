import os

from kubeflow_pipeline.utils import KFPClientManager
from kubeflow_pipeline.pipeline import launch
from dotenv import load_dotenv

# Done for ease of local testing with make command
load_dotenv()


def test_client():
    deploykf_host = os.environ["DEPLOYKF_HOST"]
    # deploykf_namespace = os.environ["DEPLOYKF_NAMESPACE"]
    deploykf_username = os.environ.get("DEPLOYKF_USER")
    deploykf_password = os.environ.get("DEPLOYKF_PASSWORD")
    os.environ["DEPLOYKF_EXPERIMENT"] = "test"
    os.environ["DEPLOYKF_RUN"] = "test"
    os.environ["PIPELINE_NAME"] = "hello_world"
    os.environ["MESSAGE"] = "hello_world from test"
    launch()
