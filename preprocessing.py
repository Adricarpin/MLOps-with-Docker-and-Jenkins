import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set path for the input
RAW_DATA_DIR = os.environ["RAW_DATA_DIR"]
RAW_DATA_FILE = os.environ["RAW_DATA_FILE"]
raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Read dataset
df = pd.read_csv(raw_data_path, sep=",")



# Income to binary
df['income'].replace(['<=50K','>50K'],[0,1], inplace=True) 

# Drop useless variables
df.drop('fnlwgt', axis=1, inplace=True)
df.drop('education.num', axis=1, inplace=True)

# Removing rows with missing data
df = df.loc[ (df['workclass'] != '?') & (df['occupation'] != '?') & (df['native.country']!= '?')]



# Split into dependend and independent variables
X = df.drop('income', axis=1)
y = df['income'].to_frame()

# Split X into continous variables and categorical variables
X_continous  = X[['age', 'capital.gain', 'capital.loss', 'hours.per.week']]

X_categorical = X[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race',
                   'sex', 'native.country']]

# Get the dummies
X_encoded = pd.get_dummies(X_categorical)

# Concatenate data
data = pd.concat([y, X_continous, X_encoded],axis=1)

# Split into train and test
train, test = train_test_split(data, test_size=0.3, stratify=data['income'])


# Set path to the outputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# Save csv
train.to_csv(train_path, index=False)
test.to_csv(test_path,  index=False)