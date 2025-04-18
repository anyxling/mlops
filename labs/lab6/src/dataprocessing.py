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


def load_data():
    # fetch dataset 
    dry_bean = fetch_ucirepo(id=602) 
    
    # data (as pandas dataframes) 
    X = dry_bean.data.features 
    y = dry_bean.data.targets
    return X, y

def preprocess_data(X, y):
    le = LabelEncoder()
    y = y.to_numpy().ravel()
    y_encoded = le.fit_transform(y)
    y_encoded 

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test