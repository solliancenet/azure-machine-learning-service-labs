# Lab 5 - Deep Learning

In this lab you train deep learning models built with Keras and a Tensorflow backend that utilize GPUs with the Azure Machine Learning service.

## Exercise 0 - Get the lab files
If you have not cloned this repository to your local machine, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio-code`.

## Exercise 1 - Get oriented to the lab files
1. On your local computer expand the folder `05-deep-learning`.
2. To run a lab, open Visual Studio Code and open the file `05_deep_learning.py`.
3. Confirm that your have setup `azure_automl` as your interpreter.
4. `05_deep_learning.py` is the Python file you will step thru executing in this lab.
5. For each step click on `Run Cell` just above the step. 

## Exercise 2 - Train an autoencoder using GPU
1. Start with Step 1. Here you will use Keras to define an autoencoder. Don't get hung up on the details of constructing the auto-encoder. The point of this lab is to show you how to train neural networks using GPU's. Execute Step 1. In the output, verify that `K.tensorflow_backend._get_available_gpus()` returned an entry describing a GPU available in your environment.
2. Once you have your autoencoder model structured, you need to train the the underlying neural network. Training this model on regular CPU's will take hours. However, you can execute this same code in an environment with GPU's for better performance. Execute Step 2. How long did your training take?
3. With a trained auto-encoder in hand, try using the model by selecting and executing Step 3.

## Exercise 3 - Register the neural network model with Azure Machine Learning
1. In Step 4, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments to create or retrieve your workspace. Observe that you can register a neural network model with Azure Machine Learning in exactly the same way you would register a classical machine learning model. Execute Step 4 to register the model.
