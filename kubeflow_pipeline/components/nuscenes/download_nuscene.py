from typing import Dict

from kfp import dsl
from kfp.dsl import Dataset


@dsl.component(base_image="python:3.10")
def download_nuscene(url: str) -> Dataset:
    import urllib.request
    import tarfile

    import os
    from pathlib import Path
    
    output_dataset = Dataset(name="nuscene", uri=dsl.get_uri(), metadata={})
    dataset_path = Path(output_dataset.path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, f"{dataset_path}/data.zip")
    tar = tarfile.open(dataset_path / "data.tgz")
    tar.extractall()
    tar.close()
jh
    os.remove(dataset_path / "data.tgz")
    
    return output_dataset
