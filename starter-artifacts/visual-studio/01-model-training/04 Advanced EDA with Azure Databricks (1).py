# Databricks notebook source
# MAGIC %md
# MAGIC #Advanced EDA with Azure Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In order to run this notebook you should have previously run the <a href="$./03 Basic EDA with Azure Databricks">Basic EDA with Azure Databricks</a> notebook to have everything prepared for this step.

# COMMAND ----------

# MAGIC %md
# MAGIC You are now done with exploring the dataset feature by feature, which is the main block in an EDA.   
# MAGIC The slightly more advanced section consist of four parts:
# MAGIC 
# MAGIC * Creating a simple baseline model (the parsimonious model)
# MAGIC * One hot encoding and feature scaling
# MAGIC * Dimensionality reduction
# MAGIC * Estimate feature importance by training a random forest regressor
# MAGIC 
# MAGIC Since this is a lot of new material that we have not covered in depth yet we have done most of the coding for you. Your job is then to evaluate and understand the results.  

# COMMAND ----------

# MAGIC %md ###Creating a simple baseline model (the parsimonious model)

# COMMAND ----------

# MAGIC %md Load the clean version of the data.
# MAGIC 
# MAGIC Be sure to update the table name  "usedcars\_clean\_#####" with the unique name created while running the <a href="$./02.03 Basic EDA with Azure Databricks">Basic EDA with Azure Databricks</a> notebook.

# COMMAND ----------

import numpy as np
import pandas as pd

df = spark.sql("SELECT * FROM usedcars_clean_#####")

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
# MAGIC The following cell will create a new Spark DataFrame that has our two desired features and the engineered label.

# COMMAND ----------

df_affordability = df.selectExpr("Age","KM", "CASE WHEN Price < 12000 THEN 1 ELSE 0 END as Affordable")
display(df_affordability)

# COMMAND ----------

# MAGIC %md 
# MAGIC While we could use matplotlib or ggplot to create a scatter plot of our data, the Azure Databricks notebook has a built in way for us to plot the data from the DataFrame without any material code, just by calling `display()` and passing it the DataFrame. 
# MAGIC 
# MAGIC We've already configured the plot, so you just need to run the next cell. If you are curious as to the settings we used, select the Plot Options button that appears underneath the chart. 

# COMMAND ----------

display(df_affordability)

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge #1**
# MAGIC 
# MAGIC Given the above chart, at approximately what age does it look we start to afford a car irrespective of it's distance driven?

# COMMAND ----------

# MAGIC %md **Training the classifier**
# MAGIC 
# MAGIC In this particular case, we have chosen to train our classifier using the LogisticRegression module from SciKit Learn, since it's a good starting point for a model, especially when our data is not too large. 
# MAGIC 
# MAGIC The LogisticRegression module does not understand Spark DataFrames natively. Given our small dataset, one option is to collect the data on to the driver node and then process represent using arrays. The following converts our Spark DataFrame into a Pandas DataFrame. Then the features (Age and KM) are stored in the X array and the labels (Affordability are stored in the y array).

# COMMAND ----------

X = df_affordability.select("Age", "KM").toPandas().values
y = df_affordability.select("Affordable").toPandas().values

# COMMAND ----------

# MAGIC %md Run the next two cells to get a quick look at the resulting arrays:

# COMMAND ----------

X

# COMMAND ----------

y

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

# we create an instance of Neighbours Classifier and fit the data.
clf.fit(X_scaled, y)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a trained model, let's examine a feature of Azure Databricks notebooks that can help us play with the inputs to our model- widgets. 
# MAGIC 
# MAGIC When you run the following cell, two new text inputs will appear near the top of this notebook. When you edit their value move out of the input field, any cells that depend on that widget's value will be automatically re-run. 
# MAGIC 
# MAGIC For now, run the following cell and observe the Age and Distance Driven widgets that appear. Notice they have been defaulted to Age of 40 months and Distance Driven of 40000 KM. 

# COMMAND ----------

dbutils.widgets.text("Age","40", "Age (months)")
dbutils.widgets.text("Distance Driven", "40000","Distance Driven (KM)")

# COMMAND ----------

# MAGIC %md
# MAGIC Now run the following cell. It will take as input the values you specified in the widgets, scale the values and then use our classifier to predict the affordability. 

# COMMAND ----------

age = int(dbutils.widgets.get("Age"))
km = int(dbutils.widgets.get("Distance Driven"))

scaled_input = scaler.transform([[age, km]])
  
prediction = clf.predict(scaled_input)

print("Can I afford a car that is {} month(s) old with {} KM's on it?".format(age,km))
print("Yes (1)" if prediction[0] == 1 else "No (1)")

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

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge #3**
# MAGIC 
# MAGIC What grade would you give your model based on this score alone? Assume an A is 90% or better, a B is 80%-90% and so on.

# COMMAND ----------

# MAGIC %md ###One hot encoding and feature scaling

# COMMAND ----------

# MAGIC %md
# MAGIC Until now we have not encoded the feature FuelType, but before we can use this feature as input to a model or a dimensionality reduction we need to apply one hot encoding. In Machine Learning literature, one hot encoding is defined as an approach to encode categorical integer features using a one-hot aka one-of-K scheme. In a nutshell, every distinct value of the categorical integer feature becomes a new column which has all zero values except for rows where that value is present, where it has a value of 1. This is a way to transform categorical values into a form that can be more efficiently used by Machine Learning algorithms.
# MAGIC 
# MAGIC Running the next cell will store an encoded version of the dataset in a new dataframe called `df_ohe`.

# COMMAND ----------

df_ohe = df.toPandas().copy(deep=True)
df_ohe['FuelType'] = df_ohe['FuelType'].astype('category')
df_ohe = pd.get_dummies(df_ohe)

df_ohe.head(15)

# COMMAND ----------

# MAGIC %md
# MAGIC To be prepared for any model in the modelling phase, we also make a scaled dataset.    
# MAGIC The code below makes a new dataframe called `df_ohe_scaled` 

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
columns_to_scale = ['Age', 'KM', 'HP', 'CC','Weight']
df_ohe_scaled = df_ohe.dropna().copy()
df_ohe_scaled[columns_to_scale] = scaler.fit_transform(df_ohe.dropna()[columns_to_scale])

df_ohe_scaled.head(15)

# COMMAND ----------

# MAGIC %md ###Dimensionality reduction
# MAGIC 
# MAGIC Dimensionality rediction is the operation that transforms data with n dimensions (in pandas world n columns in the dataframe) to a representation of the data in m dimensions. Obviously m is less than n, and for visualizations we set m to be 2 or 3. 
# MAGIC 
# MAGIC To reduce the dimensionality of our dataset we use a method called Principal Component Analysis (PCA). With this method we can reduce the dimensionality in a way that preserves as much variance as possible. 
# MAGIC 
# MAGIC You can play around with the selection of features to see which features affect the PCA.   
# MAGIC You can also try the PCA using the dataframe we didn't scale to see how scale affects the transformation. 
# MAGIC 
# MAGIC What makes PCA interesting in the context of an EDA is that we can use it to explore the relationship between higher dimensional data and a respons variable. 
# MAGIC 
# MAGIC Below we send all features (not price) to the PCA to transform it to two dimensions. When we plot the two dimensional data and color it with price we get a graphical representation of the relationship between Price and all the features combined. 

# COMMAND ----------

from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()

features = ['Age', 'KM', 'HP', 'Weight', 'CC', 'Doors',  'Automatic', 'MetColor', 'FuelType_cng', 'FuelType_diesel', 'FuelType_petrol']

x_2d = PCA(n_components=2).fit_transform(df_ohe_scaled[features])
sc = plt.scatter(x_2d[:,0], x_2d[:,1], c=df_ohe_scaled['Price'], s=10, alpha=0.7)
plt.colorbar(sc) 

display(fig)

# COMMAND ----------

# MAGIC %md ###Estimate feature importance by training a random forest regressor

# COMMAND ----------

# MAGIC %md The model Random Forest has a very valuable side-product. After training the model it can provide a list over all features ranked by importance (we will bump into this concept again later in the workshop). By running the cell below you get one of these feature importance rankings. 
# MAGIC 
# MAGIC __Question:__ Does the output with feature importance match what you experienced when exploring the dataset?

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

fig, ax = plt.subplots()

features_RFR = ['Age', 'KM', 'HP', 'Weight', 'CC', 'Doors', 'Automatic', 'MetColor', 'FuelType_cng', 'FuelType_diesel', 'FuelType_petrol']

# Create train and test data
X = df_ohe[features_RFR].as_matrix()
y = df.toPandas()['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =0)

# Initialize  a random forest regressor
# 'Train' the model
RandomForestReg = RandomForestRegressor()
RandomForestReg.fit(X_train, y_train)

imp = pd.DataFrame(
        RandomForestReg.feature_importances_ ,
        columns = ['Importance'] ,
        index = features_RFR
    )
imp = imp.sort_values( [ 'Importance' ] , ascending = True )
imp['Importance'].plot(kind='barh')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This concludes the Exploratory Data Analysis lab.
# MAGIC 
# MAGIC In this lab you investigated a dataset with sale prices in $ for used (second-hand) Toyota Corollas.   
# MAGIC During the lab you used a lot of the techniques we introduced in the presentation about EDA (Exploratory data analysis).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Answers to Challenges
# MAGIC 
# MAGIC 1. Somewhere between 40 and 50 months in age.
# MAGIC 2. The scaled range for KM is -1.83 to 4.65. 
# MAGIC 3. The percentage score is 92.6%, so this would get an A.
