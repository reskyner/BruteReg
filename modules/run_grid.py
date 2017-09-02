import sys, re
sys.path.append('./modules')

import pipemodules as pm
import numpy as np
import pandas as pd


def auto_grid(X, y, labels, train_percentage, opts, input_params, sig_time, ks=range(5,100,5)):
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
                    method_id = str([j,k,i])
                    grid_search = pm.search_random_forest()
                    grid_search.set_method(i)

                    grid_search.read_method_paramgrid(input_params, i)

                    # run current method
                    temp_results = grid_search.run(X_temp, all_data.Y_train, method_id, sig_time)

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
