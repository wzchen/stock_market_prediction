# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import math
from math import log
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from time import gmtime, strftime
import scipy
import sys
import sklearn.decomposition
from sklearn.metrics import mean_squared_error
from string import punctuation
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
import time
from scipy import sparse
from matplotlib import *
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import operator

# <codecell>

def tied_rank(x):
    """
    This function is by Ben Hamner and taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py

    Computes the tied rank of elements in x.

    This function computes the tied rank of elements in x.

    Parameters
    ----------
    x : list of numbers, numpy array

    Returns
    -------
    score : list of numbers
            The tied rank f each element in x

    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i): 
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1): 
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def auc(actual, posterior):
    """
    This function is by Ben Hamner and taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC error metric for binary classification.

    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.

    Returns
    -------
    score : double
            The mean squared error between actual and posterior

    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    sys.stdout.write('.')
    return auc

def auc_scorer(estimator, X, y):
    predicted = estimator.predict_proba(X)[:,1]
    return auc(y, predicted)
                
def normalize10day(stocks):
    def process_column(i):
        if operator.mod(i, 5) == 4:
            return np.log(stocks[:,i] + 1)
        else:
            return stocks[:,i] / stocks[:,0]
    n = stocks.shape[0]
    stocks_dat =  np.array([ process_column(i) for i in range(31)]).transpose()
    return stocks_dat
    

# <codecell>

print "loading data.."
train = np.array(p.read_table('./training.csv', sep = ","))
test = np.array(p.read_table('./test.csv', sep = ","))

################################################################################
# READ IN THE TEST DATA
################################################################################
# all data from opening 1 to straight to opening 10
X_test = normalize10day(test[:,range(17, 48)]) # load in test data

#X_test = X_test_stockdata

#np.identity(94)[:,range(93)]

################################################################################
# READ IN THE TRAIN DATA
################################################################################
n_windows = 490
windows = range(n_windows)

X_windows = [train[:,range(16 + 5*w, 47 + 5*w)] for w in windows]
X_windows_normalized = [normalize10day(w) for w in X_windows]
X = np.vstack(X_windows_normalized)
#X_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(n_windows)))

#X = np.hstack((X_stockindicators, X_stockdata))
#X = X_stockdata

# read in the response variable
y_stockdata = np.vstack([train[:, [46 + 5*w, 49 + 5*w]] for w in windows])
y = (y_stockdata[:,1] - y_stockdata[:,0] > 0) + 0


X_test = X_test[:,[0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]]
X = X[:,[0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]]
print "this step done"

# <codecell>

# BEST IS 133
model_ridge = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=9081)
model_randomforest = RandomForestClassifier(n_estimators = 200)

pred_ridge = []
pred_randomforest = []
new_Y = []
for i in range(10):
    indxs = np.arange(i, X.shape[0], 10)
    indxs_to_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], 10)))
    pred_ridge = pred_ridge + list(model_ridge.fit(X[indxs_to_fit,:], y[indxs_to_fit,:]).predict_proba(X[indxs,:])[:,1])
    pred_randomforest = pred_randomforest + list(model_randomforest.fit(X[indxs_to_fit,:], y[indxs_to_fit,:]).predict_proba(X[indxs,:])[:,1])                               
    new_Y = new_Y + list(y[indxs,:])
                                                                   
new_X = np.hstack((np.array(pred_ridge).reshape(len(pred_ridge), 1), np.array(pred_randomforest).reshape(len(pred_randomforest), 1)))
print new_X
new_Y = np.array(new_Y).reshape(len(new_Y), 1)

# <codecell>

model_stacker = lm.LogisticRegression()
print np.mean(cross_validation.cross_val_score(model_stacker, new_X, new_Y.reshape(new_Y.shape[0]), cv=5, scoring = auc_scorer))

# <codecell>

model_stacker.fit(new_X, new_Y.reshape(new_Y.shape[0]))

print "prediction"
# do a prediction and save it
pred_ridge_test = model_ridge.fit(X, y).predict_proba(X_test)[:,1]
pred_randomforest_test = model_randomforest.fit(X, y).predict_proba(X_test)[:,1]

new_X_test = np.hstack((np.array(pred_ridge_test).reshape(len(pred_ridge_test), 1), np.array(pred_randomforest_test).reshape(len(pred_randomforest_test), 1)))

# <codecell>

pred = model_stacker.predict_proba(new_X_test)[:,1]
testfile = p.read_csv('./test.csv', sep=",", na_values=['?'], index_col=[0,1])

# submit as D multiplied by 100 + stock id
testindices = [100 * D + StId for (D, StId) in testfile.index]

pred_df = p.DataFrame(np.vstack((testindices, pred)).transpose(), columns=["Id", "Prediction"])
pred_df.to_csv('./predictions/' + 'stacker' + '/' + 'stacker' + ' ' + strftime("%m-%d %X") + ".csv", index = False)

print "submission file created"

# <codecell>

model_stacker.coef_

