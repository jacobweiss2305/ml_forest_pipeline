from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

def split_data(feature_table: pd.DataFrame, response: str) -> Tuple:
    """Split data into traing and test set.

    Args:
        feature_table (pd.DataFrame): ready to model feature table
        response (str): response variable name

    Returns:
        Tuple: test and train data
    """    
    
    exclusions = [response]
    feature_names = [i for i in list(feature_table) if i not in exclusions]
    
    X = feature_table[feature_names]
    y = feature_table[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test

def random_over_sample(x_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple:
    """Randomly over sample the minority class.

    Args:
        x_train (pd.DataFrame): Training covariate matrix
        y_train (pd.DataFrame): Response variable

    Returns:
        Tuple: Resampled training sets.
    """    
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    return X_resampled, y_resampled

def train_model(X_resampled: pd.DataFrame, 
                y_resampled: pd.Series, 
                scikit_model_object: object):
    """Train model with scikit learn object.

    Args:
        X_resampled (pd.DataFrame): resampled covariates
        y_resampled (pd.Series): resamples response
        scikit_model_object (object): scikit learn model object
            e.g. LinearRegression()
        **kwargs: 

    Returns:
        object: fitted model
    """
    return scikit_model_object().fit(X_resampled, y_resampled)

def eval_accuracy(y_test: np.array, y_pred: np.array) -> float:
    return accuracy_score(y_test, y_pred)

def classification_report(y_test: np.array, y_pred: np.array) -> pd.DataFrame:
    """Classification report

    Args:
        y_test (np.array): hold out test data
        y_pred (np.array): model predictions

    Returns:
        pd.DataFrame: classification report
    """    
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report)
    
def forest_feat_importance_plot(model_fit: object, feature_table: pd.DataFrame) -> plt.subplot:
    """Forest based feature importance plot

    Args:
        model_fit (object): scikit learn model object (post fit)
        feature_table (pd.DataFrame): feature table

    Returns:
        plt.subplot: matplotlib plot
    """    
    feature_names = [i for i in list(feature_table) if "response" not in i]
    importances = model_fit.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_fit.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    return fig.tight_layout()

def generate_hyperparameters(level=None) -> Dict:
    """Generate hyperparameter space based on aggresive levels:
           low -> small hyperparameter space
           medium -> medium hyperparameter space
           large -> large hyperparameter space
    Args:
        how_aggresive (str): _description_

    Returns:
        Dict: _description_
    """
    aggresive_levels = ["low", "medium", "high"]
    if not level:
        level = "low"
        
    levels_mapping = {"low":{"n_estimators": [10, 100, 10],
                              "max_features": ['auto', 'sqrt'],
                              "max_depth": [10, 100, 4],
                              "min_samples_split": [2],
                              "min_samples_leaf": [2],
                              "bootstrap": [True, False]
                              }, 
                      "medium":{"n_estimators": [10, 1000, 100],
                              "max_features": ['auto', 'sqrt'],
                              "max_depth": [10, 1000, 4],
                              "min_samples_split": [2, 5],
                              "min_samples_leaf": [1, 2],
                              "bootstrap": [True, False]
                              }, 
                      "high":{"n_estimators": [10, 100, 10],
                              "max_features": ['auto', 'sqrt'],
                              "max_depth": [10, 10000, 4],
                              "min_samples_split": [2],
                              "min_samples_leaf": [2],
                              "bootstrap": [True, False]
                              }
                      }
    level_cfg = levels_mapping[level]
            
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = level_cfg["n_estimators"][0], 
                                                stop = level_cfg["n_estimators"][1],
                                                num = level_cfg["n_estimators"][2])]
    
    # Number of features to consider at every split
    max_features = level_cfg["max_features"]
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(level_cfg["max_depth"][0], 
                                             level_cfg["max_depth"][1], 
                                             num = level_cfg["max_depth"][2])]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = level_cfg["min_samples_split"]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = level_cfg["min_samples_leaf"]
    
    # Method of selecting samples for training each tree
    bootstrap = level_cfg["bootstrap"]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    return random_grid