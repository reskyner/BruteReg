#!/usr/bin/python
# coding=utf8

import time
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import pandas as pd
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV


from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm

import random

class preprocess(object):

    def __init__(self):
        None

    def data_split(self, X, y, labels, train_percentage):
        ## list the names of descriptors, and separate descriptors (X) and logS (y)
        values_old = list(X.columns.values)
        X2 = np.array(X)
        #y2 = np.array(exp_logS)
        y2 = np.array(y)

        ## Randomly split the data in half for train and test sets
        temp = random.sample(xrange(0, len(X2)), len(X2)) # generate a random order of numbers for the length of the data
        half_length = int(round((float(len(X2)) / 100) * float(train_percentage)))
        print('Using ' + str(half_length) + ' datapoints (' + str(train_percentage) + '%) for training set')
        train_ind = temp[0:half_length] # first half of random numbers for training
        test_ind = temp[half_length:] # second half of random numbers for testing

        X_train = [] 
        X_test = []
        Y_train = []
        Y_test = []
        refs_train = []
        refs_test=[]

        ## Create the training and testing sets
        for i in range(0,len(train_ind)):
            j = train_ind[i]
            X_train.append(X2[j,:])
            Y_train.append(y2[j])
            refs_train.append(labels[j]) # refcodes for the training structures

        for i in range(0,len(test_ind)):
            j = test_ind[i]
            X_test.append(X2[j,:])
            Y_test.append(y2[j])
            refs_test.append(labels[j]) # refcodes for the testing structures
            
        descriptor_matrix_train = pd.DataFrame()
        descriptor_matrix_test = pd.DataFrame()

        for i in range(0,len(train_ind)):
            descriptor_matrix_train = descriptor_matrix_train.append(X.iloc[train_ind[i],:])
            
        for i in range(0,len(test_ind)):
            descriptor_matrix_test = descriptor_matrix_test.append(X.iloc[test_ind[i],:])

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.descriptor_matrix_train = descriptor_matrix_train
        self.descriptor_matrix_test = descriptor_matrix_test


class feature_selector(object):

    def __init__(self):
        None

    def run(self, data, k_vals):

        selection_labels = []
        indvalues = []

        def select_features(data, no_k):

            k_best = feature_selection.SelectKBest(f_regression, k=no_k)
            selection = k_best.fit(data.X_train,data.Y_train)
            indicies_fr = selection.get_support(indices=True)
            
            k_best = feature_selection.SelectKBest(mutual_info_regression, k=no_k)
            selection = k_best.fit(data.X_train,data.Y_train)
            indicies_MIR = selection.get_support(indices=True)

            forest = ensemble.ExtraTreesRegressor(n_estimators=250,
                                          random_state=0)
            forest.fit(data.X_train, data.Y_train)
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                         axis=0)
            indicies_forest = np.argsort(importances)[::-1]
            indicies_forest = indicies_forest[:no_k]
            

            selection_labels = ['f_regression','mutual_info_regression','ExtraTreesRegressor']
            indvals = [indicies_fr, indicies_MIR, indicies_forest]
            
            return indvals, selection_labels
        
        for i in range(0,len(k_vals)):
    
            indvalues_temp, selection_labels_temp = select_features(data, k_vals[i])
    
            indvalues.append(indvalues_temp)
            selection_labels.append(selection_labels_temp)

        return selection_labels, indvalues

class search_random_forest(object):

    def __init__(self):
        self.results = ''
        self.method_str = ''
        self.method_no = ''
        self.clf = ''
        self.parameters = ''
        

    def set_method(self, method):
        """ set_method(method) method = 1: RandomForestRegressor, 2: ExtraTreesRegressor"""
        
        if method == 1:
            self.method_str = 'RandomForestRegressor'
            self.method_no = 1
            self.clf = ensemble.RandomForestRegressor()
        elif method == 2:
            self.method_str = 'ExtraTreesRegressor'
            self.method_no = 2
            self.clf = ensemble.ExtraTreesRegressor()
        elif method == 3:
            self.method_str = 'LinearRegression'
            self.method_no = 3
            self.clf = linear_model.LinearRegression()
        elif method == 4:
            self.method_str = 'Ridge'
            self.method_no = 4
            self.clf = linear_model.Ridge()
        elif method == 5:
            self.method_str = 'RidgeCV'
            self.method_no = 5
            self.clf = linear_model.RidgeCV()
        elif method == 6:
            self.method_str = 'Lasso'
            self.method_no = 6
            self.clf = linear_model.Lasso()
        elif method == 7:
            self.method_str = 'LassoCV'
            self.method_no = 7
            self.clf = linear_model.LassoCV()
        elif method == 8:
            self.method_str = 'LassoLarsCV'
            self.method_no = 8
            self.clf = linear_model.LassoLarsCV()
        elif method == 9:
            self.method_str = 'LassoLarsIC'
            self.method_no = 9
            self.clf = linear_model.LassoLarsIC()
        elif method == 10:
            self.method_str = 'ElasticNet'
            self.method_no = 10
            self.clf = linear_model.ElasticNet()
        elif method == 11:
            self.method_str = 'ElasticNetCV'
            self.method_no = 11
            self.clf = linear_model.ElasticNetCV()
        elif method == 12:
            self.method_str = 'svr'
            self.method_no = 12
            self.clf = svm.LinearSVR()

        return self.method_str, self.clf, self.method_no


    def set_parameters(self, *param_grid):
        """ set_params(param_grid) param_grid = dictionary of paramaters or none for default options"""
        self.parameters = []
        if param_grid:
            self.parameters = param_grid
            
        elif 1<= self.method_no <= 2:
                param_grid = \
                    {'n_estimators': range(10, 110, 10),
                     'max_features': ['auto', 'sqrt', 'log2'],
                     'criterion': ['mse', 'mae'],
                     #'min_samples_split': range(2, 50),
                     #'min_samples_leaf': range(2, 50),
                     'min_weight_fraction_leaf': np.arange(0.1, 0.6, 0.1),
                     'bootstrap': ['False', 'True'],
                     'oob_score': ['False', 'True']}
             
        elif self.method_no == 3:
                param_grid = \
                           {'fit_intercept': [True, False],
                            'normalize': [True, False]}


        elif self.method_no == 4:
                param_grid = \
                           {'fit_intercept': [True, False],
                            'normalize': [True, False],
                            'alpha': range(0, 105, 5),
                            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
                
        elif self.method_no == 5 :
                param_grid = \
                           {'alphas': range(0, 101),
                            'fit_intercept': [True, False],
                            'normalize': [True, False],
                            'cv': range(0,11),
                            'gcv_mode': ['None', 'auto', 'svd', 'eigen']}

        elif self.method_no == 6 :
                param_grid = \
                           {'alpha': range(1, 20),
                            'fit_intercept': [True, False],
                            'normalize': [True, False],
                            'precompute': [True, False],
                            'selection': ['cyclic', 'random']}
  
  
        elif self.method_no == 7:
                param_grid = \
                           {'n_alphas': range(10, 210, 10),
                            'fit_intercept': [True, False],
                            'normalize': [True, False],
                            'precompute': [True, False],
                            'cv': range(0, 10),
                            'selection': ['random', 'cyclic']}
     
        elif self.method_no == 8:
                param_grid = \
                           {'max_n_alphas': range(0, 510, 10),
                            'fit_intercept': [True, False],
                            'normalize': [True, False],
                            'precompute': [True, False],
                            'cv': range(0, 10)}
  
        elif self.method_no == 9:
                param_grid = \
                           {'criterion': ['aic', 'bic'],
                            'fit_intercept': [True, False],
                            'normalize': [True, False],
                            'precompute': [True, False]}
  
        elif self.method_no == 10:
                param_grid = \
                           {'alpha': range(-100, 100, 10),
                            'l1_ratio': np.arange(0,1,0.1),
                            'fit_intercept': [True, False],
                            'normalize': [True, False]}
 
        elif self.method_no == 11:
                param_grid = \
                           {'l1_ratio': np.arange(0.1, 1.1, 0.1),
                            'n_alphas': range(10, 560, 50),
                            'fit_intercept': [True, False],
                            'normalize': [True, False]}
        elif self.method_no == 12:
            param_grid = \
                       {'C': np.arange(0.1, 1, 0.05)}

        self.parameters = param_grid
        return self.parameters


    def run(self, X, y, meth_id):

        print 'Running a grid search (CV) with ' + str(self.method_str) + str('(' + str(meth_id) + ')') + '...'
        meth_id = eval(meth_id)

        if meth_id[2] in [1,2,4,8,10,11]:
            start = time.time()
            print('Parameter grid: ' + str(self.parameters))
            runner = GridSearchCV(self.clf, self.parameters, n_jobs=-1)
        else:
            start = time.time()
            print('Parameter grid: ' + str(self.parameters))
            runner = GridSearchCV(self.clf, self.parameters, n_jobs=1, pre_dispatch=False)
        try:
            results = runner.fit(X,y)
            ids = []
            for i in range(0, len(results.cv_results_['rank_test_score'])):
                ids.append(str(meth_id))

            results.cv_results_['method_ids']=ids
            self.ranked = pd.DataFrame(results.cv_results_)
            end = time.time()
            print('Wall clock time: ' + str(end-start))
        except:
            print('Parameter grid: ' + str(self.parameters))
            runner = GridSearchCV(self.clf, self.parameters, n_jobs=1, pre_dispatch=False)
            results = runner.fit(X, y)
            ids = []
            for i in range(0, len(results.cv_results_['rank_test_score'])):
                ids.append(str(meth_id))

            results.cv_results_['method_ids'] = ids
            self.ranked = pd.DataFrame(results.cv_results_)
            end = time.time()
            print('Wall clock time: ' + str(end - start))

        return self.ranked

    def get_results(self):

        self.printout = pd.DataFrame(self.ranked)
        
        return self.printout

def get_X(X, indicies):
    state = 0
    for a in indicies:
        if state == 0:
            X_out = pd.DataFrame(X.iloc[:,a])
            state += 1
        else:
            temp = X.iloc[:,a]
            X_out = pd.concat([X_out, temp], axis=1)

    descriptor_names = list(X_out.columns.values)
    X_out = X_out.as_matrix()      

    return X_out, descriptor_names
