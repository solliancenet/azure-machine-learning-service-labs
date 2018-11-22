import numpy as np
import pandas as pd
import os


# MAGIC %md
# MAGIC Creating a simple baseline model (the parsimonious model)

# COMMAND ----------

# MAGIC %md ###Creating a simple baseline model (the parsimonious model)

# COMMAND ----------

# MAGIC %md Load the clean version of the data.
# MAGIC 

# COMMAND ----------

print("Current working directory is ", os.path.abspath(os.path.curdir))
df = pd.read_csv('UsedCars_Clean.csv', delimiter=',')
print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we will train a parsimonious model, a basic model to get a sense of the predictive capability of our data. 
# MAGIC 
# MAGIC We are going to try and build a model that can answer the question "Can I afford a car that is X months old and has Y kilometers on it, given I have $12,000 to spend?"
# MAGIC 
# MAGIC The model will respond with a 1 (Yes) or no 0 (No). 
# MAGIC 
# MAGIC In order to train a classifier, we need labels that go along with our used car features. The only features our model will be trained with are Age and KM. 
# MAGIC 
# MAGIC We will engineer the label for Affordable. Our logic will be simple, if the car costs less than $12,000 (our stated budget), then we will label that row in our data with a 1, meaning Yes it is affordable. Otherwise we will label it with a 0.
# MAGIC 
# MAGIC The following cell will create a new DataFrame that has our two desired features and the engineered label.

# COMMAND ----------

df['Affordable'] = np.where(df['Price']<12000, 1, 0)
df_affordability = df[["Age","KM", "Affordable"]]
print(df_affordability)

# COMMAND ----------

# MAGIC %md **Training the classifier**
# MAGIC 
# MAGIC In this particular case, we have chosen to train our classifier using the LogisticRegression module from SciKit Learn, since it's a good starting point for a model, especially when our data is not too large. 
# MAGIC 
# MAGIC The LogisticRegression module does not understand Spark DataFrames natively. Given our small dataset, one option is to collect the data on to the driver node and then process represent using arrays. The following converts our Spark DataFrame into a Pandas DataFrame. Then the features (Age and KM) are stored in the X array and the labels (Affordability are stored in the y array).

# COMMAND ----------


X = df_affordability[["Age", "KM"]].values
y = df_affordability[["Affordable"]].values


# COMMAND ----------

# MAGIC %md
# MAGIC Now one challenge we will face with the LogisticRegression is that it expects the inputs to be normalized. To make a long story short, if we were just to train the model using KM and Age without normalizing them to a smaller range around 0, then the model would give undue importance to the KM values because they are simply so much larger than the age (e.g., consider 80 months and 100,000 KM). 
# MAGIC 
# MAGIC To normalize the values, we use the StandardScaler, again from SciKit-Learn.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC In the next line we look at the result of scaling. The first table of output shows the statistics for the original values. The second table shows the stats for the scaled values. Column 0 is Age and column 1 is KM. 

# COMMAND ----------

print(pd.DataFrame(X).describe().round(2))
print(pd.DataFrame(X_scaled).describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge 2**
# MAGIC 
# MAGIC After scaling, what is the range of values possible for the KM feature?

# COMMAND ----------

# MAGIC %md
# MAGIC Next we will train the model. 

# COMMAND ----------

from sklearn import linear_model
# Create a linear model for Logistic Regression
clf = linear_model.LogisticRegression(C=1)

# we create an instance of Classifier and fit the data.
clf.fit(X_scaled, y)


# COMMAND ----------

# MAGIC %md
# MAGIC Now run the following cell. It will take as input the values you specified in the widgets, scale the values and then use our classifier to predict the affordability. 

# COMMAND ----------

age = 60
km = 40000

scaled_input = scaler.transform([[age, km]])
  
prediction = clf.predict(scaled_input)

print("Can I afford a car that is {} month(s) old with {} KM's on it?".format(age,km))
print("Yes (1)" if prediction[0] == 1 else "No (0)")

# COMMAND ----------

# MAGIC %md
# MAGIC Experiment with changing the values for Age and Distance Driven by editing the values in the widgets. Notice that every time you edit a value and exit the input field, the above cell is re-executed (HINT: Look at the timestamp output that appears at the bottom of the above cell).

# COMMAND ----------

# MAGIC %md
# MAGIC The above approach let's us experiment one prediction at a time. But what if we want to score a list of inputs at once? The following cell shows how we could score all of our original features to see what our model would predict.

# COMMAND ----------

scaled_inputs = scaler.transform(X)
predictions = clf.predict(scaled_inputs)
print(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can "grade" our model's performance using the accuracy measure. To do this we are effectively comparing what the model predicted versus what the label actually was for each row in our data. 
# MAGIC 
# MAGIC An easy way to do this is by using the `accuracy_score` method from SciKit-Learn. 

# COMMAND ----------

from sklearn.metrics import accuracy_score
score = accuracy_score(y, predictions)
print("Model Accuracy: {}".format(score.round(3)))

# Now experiment with different training subsets
from sklearn.model_selection import train_test_split
full_X = df_affordability[["Age", "KM"]]
full_Y = df_affordability[["Affordable"]]

def train_eval_model(full_X, full_Y,training_set_percentage):
    train_X, test_X, train_Y, test_Y = train_test_split(full_X, full_Y, train_size=training_set_percentage, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    clf = linear_model.LogisticRegression(C=1)
    clf.fit(X_scaled, train_Y)

    scaled_inputs = scaler.transform(test_X)
    predictions = clf.predict(scaled_inputs)
    score = accuracy_score(test_Y, predictions)

    return (clf, score)

# Verify AML SDK Installed
#####################################################################
import azureml.core
print("SDK Version:", azureml.core.VERSION)



# Create a workspace
#####################################################################

#Provide the Subscription ID of your existing Azure subscription
subscription_id = "e223f1b3-d19b-4cfa-98e9-bc9be62717bc"

#Provide values for the Resource Group and Workspace that will be created
resource_group = "aml-workspace-z"
workspace_name = "aml-workspace-z"
workspace_region = 'westcentralus' # eastus, westcentralus, southeastasia, australiaeast, westeurope


import azureml.core

# import the Workspace class 
from azureml.core import Workspace

ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region)

print("Workspace Provisioning complete.")


# Create an experiment and log metrics for multiple training runs
#####################################################################

from azureml.core.run import Run
from azureml.core.experiment import Experiment

# start a training run by defining an experiment
myexperiment = Experiment(ws, "UsedCars_Experiment_02")
root_run = myexperiment.start_logging()

training_set_percentage = 0.25
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

training_set_percentage = 0.5
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

# Go to the Azure Portal, find your Azure Machine Learning Workspace, select Experiments and select the UsedCars_Experiment
# Confirm you have two runs with a status of running

training_set_percentage = 0.75
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

training_set_percentage = 0.9
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

# Close out the experiment
root_run.complete()

# Go to the Azure Portal, find your Azure Machine Learning Workspace, select Experiments and select the UsedCars_Experiment

# You can also query the run history using the SDK.
# The following command lists all of the runs for the experiment
runs = [r for r in root_run.get_children()]
print(runs)

# Submit an experiment to Azure Batch AI and log metrics for multiple training runs
###################################################################################


ws = Workspace.get(name=workspace_name, subscription_id=subscription_id,resource_group=resource_group)
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

experiment_name = "UsedCars_ManagedCompute_01"

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)

from azureml.core.compute import BatchAiCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
batchai_cluster_name = "UsedCars-02"
cluster_min_nodes = 1
cluster_max_nodes = 3
vm_size = "STANDARD_DS11_V2"
autoscale_enabled = True


if batchai_cluster_name in ws.compute_targets():
    compute_target = ws.compute_targets[batchai_cluster_name]
    if compute_target and type(compute_target) is BatchAiCompute:
        print('Found existing compute target, using this compute target instead of creating:  ' + batchai_cluster_name)
else:
    print('Creating a new compute target...')
    provisioning_config = BatchAiCompute.provisioning_configuration(vm_size = vm_size, # NC6 is GPU-enabled
                                                                vm_priority = 'lowpriority', # optional
                                                                autoscale_enabled = autoscale_enabled,
                                                                cluster_min_nodes = cluster_min_nodes, 
                                                                cluster_max_nodes = cluster_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, batchai_cluster_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current BatchAI cluster status, use the 'status' property    
    print(compute_target.status.serialize())


# Expected output:
# Creating a new compute target...
# Creating
# succeeded.....
# BatchAI wait for completion finished
# Minimum number of nodes requested have been provisioned
# {'allocationState': 'steady', 'allocationStateTransitionTime': '2018-11-17T17:56:07.361000+00:00', 'creationTime': '2018-11-17T17:52:53.601000+00:00', 'currentNodeCount': 1, 'errors': None, 'nodeStateCounts': {'idleNodeCount': 0, 'leavingNodeCount': 0, 'preparingNodeCount': 1, 'runningNodeCount': 0, 'unusableNodeCount': 0}, 'provisioningState': 'succeeded', 'provisioningStateTransitionTime': '2018-11-17T17:53:59.653000+00:00', 'scaleSettings': {'manual': None, 'autoScale': {'maximumNodeCount': 3, 'minimumNodeCount': 1, 'initialNodeCount': 1}}, 'vmPriority': 'lowpriority', 'vmSize': 'STANDARD_DS11_V2'}


# Upload the dataset to the DataStore
######################################

ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)

ds.upload(src_dir='./data', target_path='used_cars', overwrite=True, show_progress=True)


# Prepare batch training script
# - See ./training/train.py
################################


# Create estimator
#####################
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--training-set-percentage': 0.3
}

est_config = Estimator(source_directory='C:\\Users\\student\\source\\repos\\01-model-training\\01-model-training\\training',
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['scikit-learn','pandas'])

# Execute the job
#################
run = exp.submit(config=est_config)
run

# Expected output:
# Run(Experiment: UsedCars_ManagedCompute_01,
# Id: UsedCars_ManagedCompute_01_1542479348250,
# Type: azureml.scriptrun,
# Status: Starting)

# Poll for job status
run.wait_for_completion(show_output=True) # value of True will display a verbose, streaming log

# Examine the recorded metrics from the run
print(run.get_metrics())