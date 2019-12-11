import json
import pandas
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model

def init():
    global model
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_name = 'anomaly-detector'
    print('Looking for model path for model: ', model_name)
    model_path = Model.get_model_path(model_name)
    print('Looking for model in: ', model_path)
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    print('Model loaded...')

def run(input_str):
    try:
        input_json = json.loads(input_str)
        input_df = pandas.DataFrame([[input_json['machine']['temperature'],input_json['machine']['pressure'],input_json['ambient']['temperature'],input_json['ambient']['humidity']]])
        pred = model.predict(input_df)
        print("Prediction is ", pred[0])
        if pred[0] == 1:
            input_json['anomaly']=True
        else:
            input_json['anomaly']=False
        result = [json.dumps(input_json)]
    except Exception as e:
        print(e)
        result = [str(e)]
    return result
