import pandas as pd

import json
import os
from joblib import dump

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

# Read data
df = pd.read_csv(train_data_path, sep=",")

# Split data into dependent and independent variables
X_train = df.drop('income', axis=1)
y_train = df['income']


# Model 
logit_model = LogisticRegression(max_iter=10000)
logit_model = logit_model.fit(X_train, y_train)

# Cross validation
cv = StratifiedKFold(n_splits=3) 
val_logit = cross_val_score(logit_model, X_train, y_train, cv=cv).mean()

# Validation accuracy to JSON
train_metadata = {
    'validation_acc': val_logit
}


# Set path to output (model)
MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)

# Serialize and save model
dump(logit_model, model_path)


# Set path to output (metadata)
RESULTS_DIR = os.environ["RESULTS_DIR"]
train_results_file = 'train_metadata.json'
results_path = os.path.join(RESULTS_DIR, train_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(train_metadata, outfile)

