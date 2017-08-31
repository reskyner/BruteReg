import time
import sys
import re
#sys.path.append('/usr/local/lib/python2.7/site-packages')

import pandas as pd
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV


from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
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
        print('\nUsing ' + str(half_length) + ' datapoints (' + str(train_percentage) + '%) for training set\n')
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

    def read_method_paramgrid(self, input_file, method_in):

        full_method_string = re.compile(str('method\s*=\s*') + str(method_in))
        method_string = re.compile('method\s*=\s*')
        parameter_file = open(input_file, 'r').readlines()

        string = []

        line_no = 0
        line_2_no = 0

        parsing = False
        for line in parameter_file:
            line_no += 1
            if '#' in line:
                continue

            if re.search(full_method_string, line):
                string = []
                if int(re.sub(method_string, '', line))==method_in:
                    hit = True
                else:
                    hit = False
            else:
                hit = False

            if hit:
                for line in parameter_file:
                    line_2_no += 1

                    if line_2_no < line_no:
                        continue

                    if '{' in line:
                        parameter_grid = {}
                        parsing = True

                    if '}' in line:
                        parsing = False
                        final_grid = parameter_grid  ## parameter grid fully assigned at this point, so do parameter grid here
                        hit = False
                        break

                    if parsing:
                        if '{' not in line:
                            string.append(line.replace('\n', ''))

                    for i in string:
                        eval(str('parameter_grid.update({' + str(i) + '})'))

        self.parameters = final_grid
        return self.parameters


    def run(self, X, y, meth_id):

        print 'Running a grid search (CV) with ' + str(self.method_str) + str('(' + str(meth_id) + ')') + '...'
        meth_id = eval(meth_id)

        if meth_id[2] in [1,2,4,5,7,8,10,11]:
            try:
                start = time.time()
                print('Parameter grid: ' + str(self.parameters))
                runner = GridSearchCV(self.clf, self.parameters, n_jobs=-1)
            except:
                print('parralel run failed, trying again...')
                start = time.time()
                print('Parameter grid: ' + str(self.parameters))
                runner = GridSearchCV(self.clf, self.parameters, n_jobs=1, pre_dispatch=False)
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
