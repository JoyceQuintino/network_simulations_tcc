import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle

def definitions_algorithms(name):
    if name == 'MLP':
        mlp_parameters = {'alpha':[0.1],
                          'activation':['logistic', 'tanh', 'relu'],
                          'learning_rate_init':[0.001],
                          'solver':['lbfgs'],
                          'hidden_layer_sizes':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        return mlp_parameters
    if name == 'DT':
        dt_parameters = {'criterion':['mae','mse'],
                         'splitter':['best']}
        return dt_parameters
    if name == 'RF':
        rf_parameters = {'criterion':['mae','mse'],
                         'n_estimators':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        return rf_parameters
    if name == 'SVR':
        svr_parameters = {'kernel':['poly', 'rbf'],
                          'gamma':[0, 10, 100],
                          'C':[1.0]}
        return svr_parameters
        
def optimize_parameters(reg, x, y, parameters):
    grid_search = GridSearchCV(reg, parameters, cv=2, refit=False)
    grid_search.fit(x, y)
    return grid_search.best_params_

#Cross-validation
def experiment_folds(st, reg, x, y):
    inter = range(1,31)
    scores = []
    errors = []
    models = {}
    mean_scores = []
    mean_errors = []

    for i in inter:
        kf = KFold(n_splits=5, random_state=i, shuffle=False)
        kf.get_n_splits(x)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            reg.fit(x_train, y_train)
            scores.append(reg.score(x_test, y_test))
            rmse = np.sqrt(mean_squared_error(y_test, reg.predict(x_test)))
            errors.append(rmse)

        mean_scores.append(np.mean(scores))
        mean_errors.append(np.mean(errors))
    
    return np.mean(mean_scores), np.mean(mean_errors), reg

#Leave one out
def experiment_folds2(st, reg, x, y):
    inter = range(1,31)
    scores = []
    errors = []
    models = {}
    mean_scores = []
    mean_errors = []
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        reg.fit(x_train, y_train)
        scores.append(reg.score(x_test, y_test))
        rmse = np.sqrt(mean_squared_error(y_test, reg.predict(x_test)))
        errors.append(rmse)
        
    mean_scores.append(np.mean(scores))
    mean_errors.append(np.mean(errors))
    
    return np.mean(mean_scores), np.mean(mean_errors), reg

def extraction_model(st, i, cam):
    filename = cam+'modelos/model_{}_{}.sav'.format(st, i)

    model = pickle.load(open(filename, 'rb'))
    return model

def confidence_interval(dataset):
    n = len(dataset)
    mean = np.mean(dataset)
    std = np.std(dataset)
    e = 1.96*(std/np.sqrt(len(dataset)))
    icmin = mean - e
    icmax = mean + e
    icmin = round(icmin, 0)
    icmax = round(icmax, 0)
    return icmin, icmax 

'''
def interval_confidence(dataset):
    n = len(dataset)
    u = np.mean(dataset)
    sigma = np.std(dataset)
    se = sigma/np.sqrt(n)

    interval = sc.stats.norm.interval(0.95, loc=u, scale=se)
    print('Intervalo de confianca - ', interval[0])
'''

def salve_model(st, model, cam):
    filename = cam+'modelos/model_{}.sav'.format(st)
    pickle.dump(model, open(filename, 'wb'))