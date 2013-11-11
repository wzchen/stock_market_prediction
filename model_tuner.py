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
from sklearn import svm

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
        if operator.mod(i, 5) == 1:
            return stocks[:,i] * 0
        if operator.mod(i, 5) == 2:
            return stocks[:,i] * 0
        if operator.mod(i, 5) == 4:
            return stocks[:,i] * 0
            #return np.log(stocks[:,i] + 1)
        else:
            return stocks[:,i] / stocks[:,0]
    n = stocks.shape[0]
    stocks_dat =  np.array([ process_column(i) for i in range(46)]).transpose()
    #stocks_movingavgO9O10 = np.array([int(i > j) for i,j in zip(stocks_dat[:,45], stocks_dat[:,40])]).reshape((n, 1))
    #stocks_movingavgC9O10 = np.array([int(i > j) for i,j in zip(stocks_dat[:,45], stocks_dat[:,43])]).reshape((n, 1))
    #return np.hstack((stocks_dat, stocks_movingavgO9O10, stocks_movingavgC9O10))
    return stocks_dat
    

# <codecell>


# <codecell>

print "loading data.."
train = np.array(p.read_table('./training.csv', sep = ","))
test = np.array(p.read_table('./test.csv', sep = ","))

################################################################################
# READ IN THE TEST DATA
################################################################################
# all data from opening 1 to straight to opening 10
X_test_stockdata = normalize10day(test[:,range(2, 48)]) # load in test data
X_test_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(25)))

#X_test = np.hstack((X_test_stockindicators, X_test_stockdata))
X_test = X_test_stockdata

#np.identity(94)[:,range(93)]

################################################################################
# READ IN THE TRAIN DATA
################################################################################
n_windows = 490
windows = range(n_windows)

X_windows = [train[:,range(1 + 5*w, 47 + 5*w)] for w in windows]
X_windows_normalized = [normalize10day(w) for w in X_windows]
X_stockdata = np.vstack(X_windows_normalized)
X_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(n_windows)))

#X = np.hstack((X_stockindicators, X_stockdata))
X = X_stockdata

# read in the response variable
y_stockdata = np.vstack([train[:, [46 + 5*w, 49 + 5*w]] for w in windows])
y = (y_stockdata[:,1] - y_stockdata[:,0] > 0) + 0

print "this step done"

# <codecell>

print "preparing models"

modelname = "lasso"

if modelname == "ridge": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty = "l2", C = c) for c in C]

if modelname == "lasso": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty = "l1", C = c) for c in C]

if modelname == "sgd": 
    C = np.linspace(0.00005, .01, num = 5)
    models = [lm.SGDClassifier(loss = "log", penalty = "l2", alpha = c, warm_start = False) for c in C]
    
if modelname == "randomforest":
    C = np.linspace(50, 300, num = 10)
    models = [RandomForestClassifier(n_estimators = int(c)) for c in C]

print "calculating cv scores"
cv_scores = [0] * len(models)
for i, model in enumerate(models):
    # for all of the models, save the cross-validation scores into the array cv_scores
    cv_scores[i] = np.mean(cross_validation.cross_val_score(model, X, y, cv=5, scoring = auc_scorer))
    #cv_scores[i] = np.mean(cross_validation.cross_val_score(model, X, y, cv=5, score_func = auc))
    print " (%d/%d) C = %f: CV = %f" % (i + 1, len(C), C[i], cv_scores[i])

# find which model and C is the best
best = cv_scores.index(max(cv_scores))
best_model = models[best]
best_cv = cv_scores[best]
best_C = C[best]
print "BEST %f: %f" % (best_C, best_cv)

print "training on full data"
# fit the best model on the full data
best_model.fit(X, y)

print "prediction"
# do a prediction and save it
pred = best_model.predict_proba(X_test)[:,1]
testfile = p.read_csv('./test.csv', sep=",", na_values=['?'], index_col=[0,1])

# submit as D multiplied by 100 + stock id
testindices = [100 * D + StId for (D, StId) in testfile.index]

pred_df = p.DataFrame(np.vstack((testindices, pred)).transpose(), columns=["Id", "Prediction"])
pred_df.to_csv('./predictions/' + modelname + '/' + modelname + ' ' + strftime("%m-%d %X") + " C-" + str(round(best_C,4)) + " CV-" + str(round(best_cv, 4)) + ".csv", index = False)

print "submission file created"

