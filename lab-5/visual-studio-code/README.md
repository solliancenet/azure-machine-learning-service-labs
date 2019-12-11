# Lab 5 - Deep Learning

In this lab you train deep learning models built with Keras and a Tensorflow backend that utilize GPUs with the Azure Machine Learning service.

## Exercise 0 - Get the lab files
Confirm that you have completed lab: [lab-0](../../lab-0/visual-studio-code-setup) for Visual Studio Code before you begin.

## Exercise 1 - Get oriented to the lab files
1. On your local computer expand the folder `05-deep-learning`.
2. To run a lab, start Visual Studio Code and open the folder: `05-deep-learning` and select the starting notebook file: `05_deep_learning.ipynb`.
3. Confirm that your have setup `azure_automl` as your interpreter.
4. `05_deep_learning.ipynb` is the notebook file you will step thru executing in this lab.
5. For each step click on `Run Cell` just above the step.

## Exercise 1 - Get oriented to the lab files
1. Within Azure Notebook VM's Jupyter Notebooks interface navigate to `starter-artifacts/nbvm-notebooks/05-deep-learning`.
2. To run a lab, open `05_deep_learning.ipynb`. This is the Python notebook you will step thru executing in this lab.

## Exercise 2 - Train an autoencoder using GPU
1. Begin with Step 1. In this step you are acquiring (or creating) an instance of your Azure Machine Learning Workspace. In this step, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments. Next, you will create or retreive your AML Compute cluster with GPU support. You can visit [GPU optimized virtual machine sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#nc-series) to learn more about available GPU optimized compute resources with Azure Machine Learning. Execute Step #1.
2. Execute Step 2 to create the autoencoder deep learning model training script that will be run remotely on the AML compute. Review the training script `train.py`. Note that the script will register the trained model with Azure Machine Learning.
3. Execute Step 3 to submit the model training job on the AML compute. Wait for the run to complete before proceeding to the next exercise.

## Exercise 3 - Download the trained model files
1. In Step 4, you can download the trained model files from the experiment run object. The model files are saved on the remote training machine, this step shows how to access the saved output files and download them locally to your computer. Execute Step #4.

## Exercise 4 - Review registered model
1. Execute Step 5 to load and review the properties of the registered model with Azure Machine Learning.
