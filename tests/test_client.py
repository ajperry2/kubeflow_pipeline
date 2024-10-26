import os
from dotenv import load_dotenv

from kubeflow_pipeline.utils import KFPClientManager

load_dotenv()


def test_client():
    deploykf_host = os.environ["DEPLOYKF_HOST"]
    deploykf_namespace = os.environ["DEPLOYKF_NAMESPACE"]
    deploykf_username = os.environ.get("DEPLOYKF_USER")
    deploykf_password = os.environ.get("DEPLOYKF_PASSWORD")
    deploykf_experiment = os.environ.get("DEPLOYKF_EXPERIMENT")
    deploykf_run = os.environ.get("DEPLOYKF_RUN")
    kfp_client_manager = KFPClientManager(
        api_url="https://" + deploykf_host + "/pipeline",
        skip_tls_verify=True,
        dex_username=deploykf_username,
        dex_password=deploykf_password,
        dex_auth_type="local",
    )
    kfp_client = kfp_client_manager.create_kfp_client()
