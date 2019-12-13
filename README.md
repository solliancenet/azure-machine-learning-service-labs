# Azure Machine Learning Service Labs

This repo contains labs that show how to use the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py). These labs will teach you how to perform the training locally as well scale out model training by using [Azure Machine Learning compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets#amlcompute). Each lab provides instructions for you to perform them using the environment of your choice - [Azure Notebook VMs](https://azure.microsoft.com/en-us/services/machine-learning/), or [Visual Studio Code](https://code.visualstudio.com/docs/setup/setup-overview).

The following labs are available:
- [Lab 0:](./lab-0/README.md) Setting up your environment. If a lab environment has not been provided for you, this lab provides the instructions to get started in your own Azure Subscription.
- [Lab 1:](./lab-1/README.md) Setup the Azure Machine Learning service from code and create a classical machine learning model that logs metrics collected during model training.
- [Lab 2:](./lab-2/README.md) Use the capabilities of the Azure Machine Learning service to collect model performance metrics and to capture model version, as well as query the experimentation run history to retrieve captured metrics.
- [Lab 3:](./lab-3/README.md) Deploying a trained model to containers using an Azure Container Instance and Azure Kubernetes Service using Azure Machine Learning.
- [Lab 4:](./lab-4/README.md) Using the automated machine learning (AutoML) capabilities within the Azure Machine Learning service to automatically train multiple models with varying algorithms and hyperparameters and then select the best performing model.
- [Lab 5:](./lab-5/README.md) Training deep learning models built with Keras and a TensorFlow backend that utilize GPU optimized Azure Machine Learning compute cluster.
- [Lab 6:](./lab-6/README.md) Learn how Azure Machine Learning service  can be used to provision an IoT Hub and an IoT Edge Device and then deploy a trained anomaly detection model to the IoT Edge device to make real-time inferences.
