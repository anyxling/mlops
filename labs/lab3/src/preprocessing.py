from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd

# fetch dataset 
dry_bean = fetch_ucirepo(id=602) 
  
# data (as pandas dataframes) 
X = dry_bean.data.features 
y = dry_bean.data.targets 

# Label encoding for target variable
le = LabelEncoder()
y = y.to_numpy().ravel()
y_encoded = le.fit_transform(y)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train = pd.DataFrame(X_train_scaled, columns=X.columns)
train["class"] = y_train

test = pd.DataFrame(X_test_scaled, columns=X.columns)
test["class"] = y_test

# Save as CSV
train.to_csv("labs/lab3/data/processed_train_data.csv", index=False)
test.to_csv("labs/lab3/data/processed_test_data.csv", index=False)