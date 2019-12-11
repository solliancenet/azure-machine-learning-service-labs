# Lab 6 - Model Deployment to IoT Edge

In this lab you deploy a trained model container to an IoT Edge device.

## Exercise 0 - Get the lab files
Confirm that you have completed lab: [lab-0](../../lab-0/azure-notebook-vm-setup) for Azure Notebook VM before you begin.

## Exercise 1 - Get oriented to the lab files
1. Within Azure Notebook VM's Jupyter Notebooks interface navigate to `starter-artifacts/nbvm-notebooks/06_deploy_to_iot_edge`.
2. To run a lab, open `06_deploy_to_iot_edge.ipynb`. This is the Python notebook you will step thru executing in this lab.

## Exercise 2 - Create or retrieve your Azure ML Workspace
In this exercise you are acquiring (or creating) an instance of your Azure Machine Learning Workspace. In this step, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments. Execute Step #1.

## Exercise 3 - Provision an IoT Hub and an IoT Edge Device
Your IoT edge device will be managed thru an IoT Hub. This IoT Hub will also be the target to where your IoT Edge device will send its telemetry. In this exercise, you will do the following:

1. Create an IoT Hub in the Resource Group.
2. Provision an Ubuntu Linux Virtual Machine that will act as your IoT Edge device.
3. Create a digital identity for your IoT Edge device.
4. Configure the IoT Edge device with the connection string that uniquely identifies it to IoT Hub.

Execute Step #2 to complete this exercise. Save the **publicIpAddress** of the IoT edge device VM for later use.

## Exercise 4 - Build the ContainerImage for the IoT Edge Module
In this exercise you will use a previously trained model using the Azure Machine Learning SDK and deploy it along with a scoring script to an image. This model will score temperature telemetry data for anomalies. In a subsequent exercise, you will deploy this module to your IoT Edge device to perform scoring on the device. Execute Step #3.

## Exercise 5 - Deploy the modules
In this exercise you will deploy 2 modules to your IoT Edge device. One is a telemetry generator that will produce simulated temperature readings and the other will be an Azure Machine Learning module that will perform anomaly detection. Execute Step #4.

## Exercise 6 - Examine the scored messages
1. From within Azure Portal, navigate to your resource group and select your IoT hub: **sl-iot-hub**. 

2. From the `Automatic Device Management` section of your IoT hub, select **IoTEdge, slEdgeDevice**.

   ![The image shows the steps to locate your registered IoT edge device.](images/01.png 'IoT edge device')

   *Note that selecting **slEdgeDevice** should open the edge device page*

3. In the `slEdgeDevice` edge device page, confirm that the two added modules **tempSensor** and **machinelearningmodule** are in **running** status. It may take about 5-10 minutes for the two new modules to appear and start running. Once you see a `RUNTIME STATUS` of `running` for all modules you can proceed.

   ![The image shows the slEdgeDevice page and the status of the two modules: tempSensor and machinelearningmodule.](images/02.png 'IoT edge device page')

4. From within Azure Notebook VM's Jupyter Notebooks interface, open a new terminal.

5. Connect to your VM via SSH, using the `publicIpAddress` you acquired previously.

    ```
    ssh azureuser@{publicIpAddress}
    ```

6. View the anomaly detection scored messages being sent by the `machinelearningmodule` by running the following command.

    ```
    iotedge logs machinelearningmodule -f
    ```

   ![The image shows the streaming log output from the machinelearningmodule.](images/03.png 'machinelearningmodule logs')
