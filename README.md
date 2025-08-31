# Kubeflow Pipelines

## Baremetal Usage

1. Run notebook `notebooks/reddit_training.ipynb`
    - Define hyperparameters that make sense for your system
2. Metrics are recorded locally and can be observed with locally running mlflow or with the `verbose=true` options, test examples are printed to `standard out`


## Kubeflow Usage

1. Upload notebook `notebooks/pipeline_management.ipynb`
2. Define environment variables
3. Run cells defining training pipeline
3. Run/Schedule pipeline



## Pipeline Description

The pipeline is ran each day. In this process this is done:

- New data is downloaded
- The current best model is downloaded and evaluated
- If the model has degraded or is not proficient, training is ran

![Pipeline GUI](./images/train_pipeline.png "Pipeline that will run in kubeflow")

At the time of writing this I only have 500 samples in training set, so a test BLEU score of 0 is expected, though I hope in the coming days it will improve.

The pipeline records metrics in mlflow and records the hyperparameters/logs/outputs of each run.

![Metrics GUI](./images/metrics.png "Metrics reported")
![Hyperparameters GUI](./images/hyperparameters.png "Hyperparameters recorded")
