#!/usr/bin/python
# coding=utf8

import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')

import pipemodules as pm
import numpy as np
import re
import pandas as pd

def auto_grid(X, y, labels, train_percentage, ks=range(5,100,5), opts=[1,2,3,4,5,6,7,8,9,10,11,12]):
    """Run a grid search... auto_grid(X, y, labels, ks=range(10,100,10) opts=[1...12])
-------------------------------------------------------------------------------
Required: X - matrix of descriptors
	      y - response values
	      labels - labels for structures
Optional: ks - array of k-values (number of features to be selected
	      opts - array of option numbers where:

        	    1: Random forest
            	2: Extra random trees
            	3: Simple OLS linear regression
            	4: Ridge regression
            	5: Ridge regression with cross validation (CV)
            		--** WARNING: Currently not working **--
            	6: Lasso (Least Absolute Shrinkage Selection Operator) regression
            	7: Lasso with CV
            	8: Lasso with least angle regression (lars) & CV
            	9: Lasso lars with information criterion (IC) - AIC/BIC
             	10: Elastic net regression
            	11: Elastic net with CV
            	12: Linear support vector regression
            	
Development: Currently only a default set of hyper-parameters are enabled...

                To change these, edit them in the set_parameters() function in
                pipemodules.py.

                We reccomend you create a backup of the original file!"""

    ## Preprocess data
    print('Running preprocessing step...')
    all_data = pm.preprocess()
    print('Splitting data for training and testing...')
    all_data.data_split(X, y, labels, train_percentage)


    print('Running descriptors through feature selector in chunks...')
    ## Feature selection - Reduce the number of descriptors to use before regression
    all_data.k_vals = ks  

    fs = pm.feature_selector()
    all_data.selection_labels, all_data.ind_values = fs.run(all_data,all_data.k_vals)
    results = []

    ## For the number of selection labels
    print('Starting grid search step...\n')
    for j in range(0,len(all_data.selection_labels)):
        for k in range(0,len(all_data.selection_labels[j])):

            k_select = all_data.k_vals[j]    
            selection = all_data.selection_labels[j][k]

            X_temp, othertemp = pm.get_X(all_data.descriptor_matrix_train, all_data.ind_values[j][k]) 

            all_data.options = opts

            for i in all_data.options:

                try:

                    # Set up method calculation object parameters
                    method_id = str([j,k,i])
                    # print('Setting up model with index: ' + method_id)
                    grid_search = pm.search_random_forest()
                    grid_search.set_method(i)
                    
                    grid_search.set_parameters()

                    # run current method
                    # print('Running grid search for current method...')
                    temp_results = grid_search.run(X_temp, all_data.Y_train, method_id)

                    # add results to table
                    try:
                        print('Recording results for current method...')
                        all_data.results = pd.concat([all_data.results, pd.DataFrame(temp_results)])
                        print('Done with current method... woohoo!\n\n')
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        all_data.results = []
                        all_data.results = pd.DataFrame(temp_results)
                        print('Done with current method... woohoo!\n\n')

                # Error handling for failed methods
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    print(str('The current method (id: ' + str(method_id) + ') has failed... Check hyperparameters\n'))
                    continue

    ## rank the results according to mean test score
    array = np.array(all_data.results['mean_test_score'])
    temp = array.argsort()[::-1]
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    all_data.results['rank_test_score'] = ranks

    ## drop method specific columns
    regex = re.compile('param')
    to_drop = []

    for i in range(0,len(all_data.results.columns)):
        if re.search(regex,all_data.results.columns[i]):
            if len(all_data.results.columns[i].split('_')) > 1:
                to_drop.append(all_data.results.columns[i])

    for i in to_drop:
        all_data.results.drop(i, axis=1, inplace=True)

    ## reorder by rank and reset index
    all_data.results.sort_values(by='rank_test_score', inplace=True)
    all_data.results.reset_index(drop=True, inplace=True)

    return all_data

