from joblib import load

from sklearn.metrics import accuracy_score

import pandas as pd

import json
import os

# model path
MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'logit_model.joblib'

model_path = os.path.join(MODEL_DIR, model_file)


# processed data path
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]

test_data_file = 'test.csv'
test_data_path = os.path.join(PROCESSED_DATA_DIR, test_data_file)




logit_model = load(model_path)

df = pd.read_csv(test_data_path, sep=",")

# split data into dependent and independent variables
X_test = df.drop('income', axis=1)
y_test = df['income']


logit_predictions = logit_model.predict(X_test)
test_logit = accuracy_score(y_test,logit_predictions)


test_metadata = {
    'test_acc': test_logit
}



RESULTS_DIR = os.environ["RESULTS_DIR"]

test_results_file = 'test_metadata.json'
results_path = os.path.join(RESULTS_DIR, test_results_file)


with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile)





