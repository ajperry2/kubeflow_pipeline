import os
from kfp import dsl

from kubeflow_pipeline import components


@dsl.pipeline
def download_nuscene(
    nuscene_email: str,
    nuscene_password: str,
    region:str
) -> dsl.Dataset:
    download_task = components.nuscenes.download_nuscene(
        nuscene_email=nuscene_email,
        nuscene_password=nuscene_password,
        region=region,
    )
    download_task.set_display_name("STEP 0: Download Data")
    return download_task.output
