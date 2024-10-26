import os
from typing import Dict

from kfp import dsl

from kubeflow_pipeline import components


@dsl.pipeline
def download_nuscene() -> dsl.Dataset:
    data_url = os.environ.get("DATA_URL", "")
    download_task = components.download_nuscene.download_nuscene(url=data_url)
    download_task.set_display_name("STEP 0: Download Data")
    return download_task.output
