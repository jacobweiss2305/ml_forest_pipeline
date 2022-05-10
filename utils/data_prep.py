from typing import Any, Dict, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
from tqdm import tqdm

def remove_sparse_columns(raw_features: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Remove columns with N% NaN coverage
    Args:
        df (pd.DataFrame): raw feature table
        coverage (float): percent of nan coverage
    Returns:
        pd.DataFrame: dense raw feature table
    """
    return raw_features.replace("nan", np.nan).replace("", np.nan).loc[:, raw_features.isnull().sum() < parameters['sparsity'] * raw_features.shape[0]]


def continuous_variables(df: pd.DataFrame, parameters=None, special_char=False) -> Tuple:
    """Scale continuous variables and provide mapping 
    Args:
        df (pd.DataFrame): dense feature table
        parameters ([type], optional): dictionary of feature and type mapping. Defaults to None.
    Returns:
        Tuple: scaled dense feature table and scalar mapping
    """
    if parameters:
        #set columns to only continuous variables
        values = [i for i, j in parameters["feature_engineering_map"].items() if j == "continuous" and i in list(df)]
    else:
        values = list(df)
    
    if special_char:   
        for col in values:
            #removing specials characters
            df[col] = df[col].astype(str).apply(lambda x: ''.join(char for char in x if char.isalnum())).astype(float)
   
    x = df[values].astype(float)
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    # Standard Scalar sets mean to 0. We can apply 0 mean imputation for NaN values.
    return pd.DataFrame(scaler.transform(x), columns=values).fillna(0), scaler


def categorical_variables(df: pd.DataFrame, parameters: Dict, encoder: Dict) -> Tuple:
    """Convert categorical variables to continuous variables
    Args:
        df (pd.DataFrame): dense raw features
        parameters ([type], optional): dictionary of feature and type mapping. Defaults to None.
        encoder (bool, optional): binary label encoder. Defaults to False.
    Returns:
        Tuple: categorical features and label mapping
    """
    param = encoder['logic']
    values = [i for i, j in parameters["feature_engineering_map"].items() if j == param and i in list(df)]
    
    if param == 'if_exists':
        return df[values].notna()
    
    elif param == "label":
        le = preprocessing.LabelEncoder()
        collect = []
        for column in tqdm(values):
            df[column] = df[column].astype(str).replace("nan", "None").fillna("None")
            le.fit(df[column])
            df[column + f'_{param}'] = le.transform(df[column])
            temp = pd.concat([pd.DataFrame(le.classes_, columns=['input']),
                                pd.DataFrame(le.transform(le.classes_), columns=['output'])], axis=1)
            temp['attribute'] = column
            collect.append(temp)
        final = df[[i for i in list(df) if f'_{param}' in i]].reset_index()
        if 'index' in list(final):
            final = final.drop(['index'], axis=1)
        # pd.concat(collect) the reverse label encoder
        return final

def datetime_variables(df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Calculate recency given date columns
    Args:
        df (pd.DataFrame): dense raw feature table
        parameters (Dict): dictionary of feature and type mapping
    Returns:
        pd.DataFrame: [description]
    """
    values = [i for i, j in parameters["feature_engineering_map"].items() if j == "recency"]
    for column in tqdm(values):
        try:
            try:
                df[column + "_days"] = pd.to_timedelta(
                    datetime.now() - pd.to_datetime(df[column], errors="coerce")).dt.days
            except TypeError:
                df[column + "_days"] = pd.to_timedelta(datetime.now(
                    tz=timezone.utc) - pd.to_datetime(df[column], errors="coerce")).dt.days
        except OverflowError:
            pass
    final = df[[i for i in list(df) if "_days" in i]]
    if "index" in list(final):
        final = final.drop(["final"], axis=1)
    return continuous_variables(final)
