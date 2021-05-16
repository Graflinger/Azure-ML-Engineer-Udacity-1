# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This projects uses a Bank marketing dataset, which contains data of bank clients and their binary responses (Yes/No) to a marketing strategy. The aim of the projects is to predict wether a customer would say Yes or No responding to the marketing campaign.

Two ways creating such a model were created using the Azure ML Pipeline

- AutoML using a Voting Esemble as best fitted model
- Hyperparameter optimizing via Hyperdrive with a logistic regression model

Both models performed very similar, with a accuracy of 91.8% from the AutoML model and 91.2% from the logistic regression model

## Scikit-learn Pipeline
The pipeline consists of the train.py file, where the dataset is imported as a tabluar dataset. Afterwards the data gets preprocessed and the logistic regression models gets trained.
The whole pipeline is orchestrated using the jupyter notebook. Inside this file, the experiments are started and the best model will later be registerd.

**What are the benefits of the parameter sampler you chose?**
The RandomParameterSampling is a great way of choosing combinations of any kind of hyperparameter. Also it's very is to implement.

**What are the benefits of the early stopping policy you chose?**
Bandit is an early termination policy based on slack factor/slack amount and evaluation interval. The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run. Using this method the computational efficeny is improved by early termination of runs with bad metrics

## AutoML
The Configuration for the AutoML run is implemented the following way:

```python
automl_settings = {
    "enable_early_stopping" : True,
    "iteration_timeout_minutes": 5,
    "max_concurrent_iterations": 4,
    "max_cores_per_iteration": -1,
    "primary_metric": 'accuracy',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations":5
}

automl_config = AutoMLConfig(experiment_timeout_minutes=60,
                             task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target=compute_target,
                             experiment_exit_score = 0.99,
                             blocked_models = ['KNN','LinearSVM'],
                             enable_onnx_compatible_models=True,
                             training_data = ds,
                             label_column_name = y_name,
                             **automl_settings
                            )
                            
```

This config resulted into a Voting Ensemble Model with the accuracy of 91.8%

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The two models are very similar in their performanc (the accuracy is almost the same). The architecture is vastly different. The hyperdrive run uses a hand written model implementation insdie the train.py file. The AutoML run on the other hand only needs the data and some congifuration and the rest is done by itself as the name already suggests.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Adding more different choices of hyperparameters or increasing the number of max_total_runs could improve the hyperdrive run.

The AutoML run could also benefit from finetuning some of the configuration like experiment_timeout_minutes or iteration_timeout_minutes
