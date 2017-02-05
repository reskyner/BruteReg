#!/usr/bin/python
# coding=utf8

#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')

import pandas as pd
from sklearn import feature_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np

def select_featurez(X_train, Y_train, no_k, descriptor_matrix_train, descriptor_matrix_test):

    k_best = feature_selection.SelectKBest(f_regression, k=no_k)
    selection = k_best.fit(X_train,Y_train)
    indicies_fr = selection.get_support(indices=True)
    X_fr_train=[]
    X_fr_test=[]
    
    for a in range(0,len(indicies_fr)):
        if a==0:
            X_fr_train=pd.DataFrame(descriptor_matrix_train.iloc[:,indicies_fr[a]])
            X_fr_test=pd.DataFrame(descriptor_matrix_test.iloc[:,indicies_fr[a]])
	if a>0:
            X_fr_train=pd.concat([X_fr_train,pd.DataFrame(descriptor_matrix_train.iloc[:,indicies_fr[a]])],axis=1)
	    X_fr_test=pd.concat([X_fr_test,pd.DataFrame(descriptor_matrix_test.iloc[:,indicies_fr[a]])],axis=1)

    values_fr = list(X_fr_train.columns.values)
    X_fr_train = np.array(X_fr_train)
    X_fr_test = np.array(X_fr_test)

    k_best = feature_selection.SelectKBest(mutual_info_regression, k=no_k)
    selection = k_best.fit(X_train,Y_train)
    indicies_MIR = selection.get_support(indices=True)
    X_MIR_train=[]
    X_MIR_test = []
    
    for b in range(0,len(indicies_MIR)):
        if b==0:
            X_MIR_train=pd.DataFrame(descriptor_matrix_train.iloc[:,indicies_MIR[b]])
            X_MIR_test=pd.DataFrame(descriptor_matrix_test.iloc[:,indicies_MIR[b]])
	if b>0:
            X_MIR_train=pd.concat([X_MIR_train,pd.DataFrame(descriptor_matrix_train.iloc[:,indicies_MIR[b]])],axis=1)
	    X_MIR_test=pd.concat([X_MIR_test,pd.DataFrame(descriptor_matrix_test.iloc[:,indicies_MIR[b]])],axis=1)

    values_MIR = list(X_MIR_train.columns.values)
    X_MIR_train = np.array(X_MIR_train)
    X_MIR_test = np.array(X_MIR_test)


    forest = ExtraTreesRegressor(n_estimators=250,
                                  random_state=0)
    forest.fit(X_train, Y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indicies_forest = np.argsort(importances)[::-1]
    indicies_forest = indicies_forest[:no_k]
    
    X_forest_train=[]
    X_forest_test=[]
    
    for c in range(0,len(indicies_forest)):
        if c==0:
            X_forest_train=pd.DataFrame(descriptor_matrix_train.iloc[:,indicies_forest[c]])
            X_forest_test=pd.DataFrame(descriptor_matrix_test.iloc[:,indicies_forest[c]])
	if c>0:
            X_forest_train=pd.concat([X_forest_train,pd.DataFrame(descriptor_matrix_train.iloc[:,indicies_forest[c]])],axis=1)
	    X_forest_test=pd.concat([X_forest_test,pd.DataFrame(descriptor_matrix_test.iloc[:,indicies_forest[c]])],axis=1)

    values_forest = list(X_forest_train.columns.values)
    X_forest_train = np.array(X_forest_train)
    X_forest_test = np.array(X_forest_test)
    
    pro_train_X = [X_fr_train, X_MIR_train, X_forest_train]
    pro_test_X = [X_fr_test, X_MIR_test, X_forest_test]
    con_values = [values_fr, values_MIR, values_forest]
    selection_labels = ['f_regression','mutual_info_regression','ExtraTreesRegressor']
    
    return pro_train_X, pro_test_X, con_values, selection_labels # con_values, selection_labels
