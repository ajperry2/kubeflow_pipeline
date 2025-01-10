# Kubeflow Pipelines


## Install Package


`poetry install`

## Run Pipeline

**Run with local command**
```
poetry run kubeflow-pipeline \
    --host deploykf.example.com \
    --namespace my-namespace \
    --experiment test \
    --username 'pipeline_username' \
    --password 'pipeline_password' \ 
    --pipeline hello_world \ 
    --args '{"message": "Goodbye World!"}'
```

**Pipeline that will run in kubeflow**

![Pipeline GUE](./images/test_pipeline_run.png "Pipeline that will run in kubeflow")