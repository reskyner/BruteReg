import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')

from sklearn import metrics

## import modules to build pipelines
import projecthandle as proj
import run_grid as rg

import pandas as pd
import numpy as np

# todo: input file, output file(s),

def quality_filter(all_data, min_train_score=0.75, max_diff=0.15):

    # todo: customisable train score and difference

    results = all_data.eval_results

    results.reset_index(drop=True, inplace=True)

    ## filter results and eliminate poor models
    for i in range(0,len(results)):
        if results.mean_train_score[i] > min_train_score         \
                and abs(results.mean_test_score[i] - results.mean_train_score[i]) < max_diff:
            continue
        else: 
            results.drop(i, axis=0, inplace=True)

    results.reset_index(drop=True, inplace=True)

    ## create analysis set
    # set arrays for results
    dev_set_score = []
    eval_set_score = []
    dev_evs = []
    eval_evs = []
    dev_mae = []
    eval_mae = []
    dev_mse = []
    eval_mse = []
    dev_medae = []
    eval_medae = []
    method_ids = []
    parameters = []


    for i in range(0,len(results)):
        ## take method_ids and build estimator for current method

        # add calculated metrics, methods, and parameters to lists for results
        dev_set_score.append(clf.score(X_dev_temp, all_data.dev_set.y_raw))
        eval_set_score.append(clf.score(X_eval_temp, all_data.eval_set.y_raw))
        dev_evs.append(metrics.explained_variance_score(dev_predict, all_data.dev_set.y_raw))
        eval_evs.append(metrics.explained_variance_score(eval_predict, all_data.eval_set.y_raw))
        dev_mae.append(metrics.mean_absolute_error(dev_predict, all_data.dev_set.y_raw))
        eval_mae.append(metrics.mean_absolute_error(eval_predict, all_data.eval_set.y_raw))
        dev_mse.append(metrics.mean_squared_error(dev_predict, all_data.dev_set.y_raw))
        eval_mse.append(metrics.mean_squared_error(eval_predict, all_data.eval_set.y_raw))
        dev_medae.append(metrics.median_absolute_error(dev_predict, all_data.dev_set.y_raw))
        eval_medae.append(metrics.median_absolute_error(eval_predict, all_data.eval_set.y_raw))
        method_ids.append(string)
        parameters.append(params)

    # create dictionary object from results
    evaluation_results = {'dev_set_score':dev_set_score, 'eval_set_score':eval_set_score,
                          'method_ids':method_ids, 'parameters':parameters, 'dev_evs':dev_evs,
                          'eval_evs':eval_evs, 'dev_mae':dev_mae, 'eval_mae':eval_mae,
                          'dev_mse': dev_mse, 'eval_mse':eval_mse, 'dev_median_ae':dev_medae,
                          'eval_median_ae':eval_medae}
    

    # re-rank and sort filtered methods by test-score (r**2)
    analysis_set = pd.DataFrame(evaluation_results)
    array = np.array(analysis_set['eval_set_score'])
    temp = array.argsort()[::-1]
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    analysis_set['rank_test_score'] = ranks
    analysis_set.sort_values(by='rank_test_score', inplace=True)
    analysis_set.reset_index(drop=True, inplace=True)
    
    return analysis_set

def save_eval(filename, all_data):

    # todo: customisable data split by percentage train - linked to pipemodules.preprocess.datasplit()

    devobj = proj.input_object()
    evalobj = proj.input_object()
    methobj = proj.method_object()
    projectobj = proj.project()
    devobj.create_object(all_data.X_train, all_data.Y_train, all_data.descriptor_matrix_train)
    evalobj.create_object(all_data.X_test, all_data.Y_test, all_data.descriptor_matrix_test)
    methobj.create_object(all_data.k_vals, all_data.selection_labels, all_data.ind_values, all_data.options)
    projectobj.save_project(all_data.results, devobj, evalobj, methobj, filename)

X,y,labels = proj.set_input('./input_files/hfe_descriptors_2.csv')
results = rg.auto_grid(X, y, labels)
proj.save_eval('./results/evaluation_set', results)
analysis_set = quality_filter(results)
proj.save_analysis('./results/analysis_set', analysis_set)

# to_drop = []
# for i in range(0, len(evaluation_results)-1):
#     if str(evaluation_results.method_ids[i])==str(evaluation_results.method_ids[i+1]):
#         to_drop.append(i+1)
#     else:
#         continue
#
# for i in to_drop:
#     evaluation_results.drop(i, inplace=True)
#
# evaluation_results.sort_values(by='rank_test_score', inplace=True)
# evaluation_results.reset_index(drop=True, inplace=True)
#
# currind = 0
# to_drop = []
# for i in range(0, len(evaluation_results)-1):
#     if abs(evaluation_results.eval_set_score[currind] - evaluation_results.eval_set_score[i+1]) < 0.01         and abs(evaluation_results.dev_set_score[i] - evaluation_results.dev_set_score[i+1]) < 0.01:
#         to_drop.append(i+1)
#     else:
#         currind = i
#
# for i in to_drop:
#     evaluation_results.drop(i, inplace=True)
#
# evaluation_results.sort_values(by='rank_test_score', inplace=True)
# evaluation_results.reset_index(drop=True, inplace=True)
#
# to_drop = []
# for i in range(0, len(evaluation_results)-1):
#     if str(evaluation_results.method_ids[i])==str(evaluation_results.method_ids[i+1]):
#         to_drop.append(i+1)
#     else:
#         continue
#
# for i in to_drop:
#     evaluation_results.drop(i, inplace=True)
#
# evaluation_results.sort_values(by='rank_test_score', inplace=True)
# evaluation_results.reset_index(drop=True, inplace=True)




