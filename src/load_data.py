"""
Adapted from
- https://github.com/zzzace2000/nodegam/blob/main/nodegam/data.py
- https://github.com/google-research/google-research/blob/master/neural_additive_models/data_utils.py
- https://github.com/zzzace2000/nodegam/blob/main/nodegam/mypreprocessor.py
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import requests
import tqdm

class OurDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)

class MinMaxPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.y_mu = 0
        self.y_std = 1

    def fit(self, X, y=None):
        self.scaler.fit(X)
    
    def transform(self, X, y=None):
        return self.scaler.transform(X), y

class OurPreprocessor:
    """
    Compared to https://github.com/zzzace2000/nodegam/blob/main/nodegam/mypreprocessor.py, 
        we replace the LeaveOneOutEncoder with OrdinalEncoder for categorical features.
        We also remove the "quantile_noise" for simplicity.
    """
    def __init__(self, random_state=0, cat_features=None,
                 y_normalize=False, output_distribution='normal', n_quantiles=2000):
        
        self.random_state = random_state
        self.cat_features = cat_features
        self.y_normalize = y_normalize
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles

        self.y_mu, self.y_std = None, None
        self.feature_names = None
        self.orienc = None
        self.qt = None

    def fit(self, X, y):

        self.feature_names = X.columns
        
        if self.cat_features is not None:
            self.orienc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.orienc.fit(X[self.cat_features])

        X_copy = X.copy()
        if self.cat_features is not None:
            X_copy[self.cat_features] = self.orienc.transform(X_copy[self.cat_features])
        
        self.qt = QuantileTransformer(random_state=self.random_state,
                                 n_quantiles=self.n_quantiles,
                                 output_distribution=self.output_distribution,
                                 copy=False)
        self.qt.fit(X_copy)
        
        if y is not None and self.y_normalize:
            self.y_mu, self.y_std = y.mean(axis=0), y.std(axis=0)

    def transform(self, X, y=None):

        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        if self.cat_features is not None:
            X[self.cat_features] = self.orienc.transform(X[self.cat_features])
        X = self.qt.transform(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float32)        
        if y is not None and self.y_mu is not None:
            y = (y - self.y_mu) / self.y_std
            y = y.astype(np.float32)
        
        return X, y

def load_mimic2(DATA_PATH = "./data", fold=0):
    
    if not os.path.exists(os.path.join(DATA_PATH, 'mimic2/mimic2.data')):
        os.makedirs(DATA_PATH, exist_ok=True)
        os.system(f"wget -O {os.path.join(DATA_PATH, 'mimic2.zip')} https://api.onedrive.com/v1.0/shares/u\!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBckhtbUZIQ1NYVElnOGRub04yRUhwREppTldramc_ZT1xZFdMTUs/root/content")
        os.system(f"unzip {os.path.join(DATA_PATH, 'mimic2.zip')} -d {DATA_PATH}")
        os.system(f"rm {os.path.join(DATA_PATH, 'mimic2.zip')}")

    cols = ['Age', 'GCS', 'SBP', 'HR', 'Temperature',
            'PFratio', 'Renal', 'Urea', 'WBC', 'CO2', 'Na', 'K',
            'Bilirubin', 'AdmissionType', 'AIDS',
            'MetastaticCancer', 'Lymphoma', 'HospitalMortality']
    
    df = pd.read_csv(os.path.join(DATA_PATH, 'mimic2/mimic2.data'), names=cols, delim_whitespace=True)
    
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1].values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic2', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic2', 'test%d.txt') % fold, header=None)[0].values

    cat_features = ['GCS', 'Temperature', 'AdmissionType', 'AIDS',
                    'MetastaticCancer', 'Lymphoma', 'Renal']
    for c in cat_features:
        X_df[c] = X_df[c].astype('string')

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx],
        'cat_features': cat_features
    }

def load_mimic3(DATA_PATH = "./data", fold=0):

    if not os.path.exists(os.path.join(DATA_PATH, 'mimic3/adult_icu.gz')):
        os.makedirs(DATA_PATH, exist_ok=True)
        os.system(f"wget -O {os.path.join(DATA_PATH, 'mimic3.zip')} https://api.onedrive.com/v1.0/shares/u\!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBckhtbUZIQ1NYVElnOGQ2bzZpY0FVTHlhMjRpeXc_ZT1zN1ROeGE/root/content")
        os.system(f"unzip {os.path.join(DATA_PATH, 'mimic3.zip')} -d {DATA_PATH}")
        os.system(f"rm {os.path.join(DATA_PATH, 'mimic3.zip')}")
    
    df = pd.read_csv(os.path.join(DATA_PATH, 'mimic3/adult_icu.gz'), compression='gzip')

    train_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian',
        'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max',
        'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min',
        'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate',
        'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']

    label = 'mort_icu'

    X_df = df[train_cols]
    y_df = df[label].values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic3', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic3', 'test%d.txt') % fold, header=None)[0].values

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx]
    }

def load_income(DATA_PATH = "./data", fold=0):

    if not os.path.exists(os.path.join(DATA_PATH, 'income/adult.data')):
        os.makedirs(DATA_PATH, exist_ok=True)
        os.system(f"wget -O {os.path.join(DATA_PATH, 'income.zip')} https://api.onedrive.com/v1.0/shares/u\!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBckhtbUZIQ1NYVElnOGQ5eGNSRzR3eE55VTJDNnc_ZT1YQ2I0UVg/root/content")
        os.system(f"unzip {os.path.join(DATA_PATH, 'income.zip')} -d {DATA_PATH}")
        os.system(f"rm {os.path.join(DATA_PATH, 'income.zip')}")
        os.system(f"mv {os.path.join(DATA_PATH, 'adult')} {os.path.join(DATA_PATH, 'income')}")
    cols = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    df = pd.read_csv(os.path.join(DATA_PATH, 'income/adult.data'), header=None)
    df.columns = cols

    X_df = df.iloc[:, :-1]

    y_df = df.iloc[:, -1].copy()
    # Make it as 0 or 1
    y_df.loc[y_df == ' >50K'] = 1.
    y_df.loc[y_df == ' <=50K'] = 0.
    y_df = y_df.values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'income', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'income', 'test%d.txt') % fold, header=None)[0].values

    cat_features = X_df.columns[X_df.dtypes == object]

    for c in cat_features:
        X_df[c] = X_df[c].astype('string')

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx],
        'cat_features': cat_features
    }

def load_housing():

    data = fetch_california_housing()
    X = data.data.copy()
    y = data.target.copy()
    feature_names = data.feature_names

    X = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=0,
        shuffle=True,
        stratify=None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1 / (0.1 + 0.7),
        random_state=0,
        shuffle=True,
        stratify=None
    )
    
    return {
        'problem': 'regression',
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }
