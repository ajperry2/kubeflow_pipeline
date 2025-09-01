# Kubeflow Pipelines

## Purpose

The purpose of this repository is a place to learn new things. From infrastructure to new LLM training techniques.

## State

Data: Currently I have a functional transformer pipeline which learns common n-grams in human speech. This is due to the lack of data primarily. 

Model: The transformer model is tested, and has been compared to other implementation. It's prediction time is sensible and small enough for my local hardware.

## Future Work

**Improved ML**: Currently only supervised learning is employed, I expect the performance will plateau without reinforcement learning with human feedback (RLHF). This is to be added to the reddit pipeline.

**Improved Logging**: After training I'd like a set of input output pairs logged to MlFlow for increased transparency to output deficiencies.



## Pipelines
1. [Reddit](#reddit) Iteratively learning to create engaging posts with reddit data.



# Reddit


## Baremetal Usage

1. Run notebook `notebooks/reddit_training.ipynb`
    - Define hyperparameters that make sense for your system
2. Metrics are recorded locally and can be observed with locally running mlflow or with the `verbose=true` options, test examples are printed to `standard out`


## Kubeflow Usage

1. Upload notebook `notebooks/reddit_pipeline.ipynb`
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
