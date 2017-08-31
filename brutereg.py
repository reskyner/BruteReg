import sys, getopt, os
import argparse
#sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')

from sklearn import metrics
import pandas as pd
import numpy as np

## import modules to build pipelines
import projecthandle as proj
import run_grid as rg


## suppress all warnings - will stop stupid convergence thing - consider revising
# todo: figure out which models auto throw warning (e.g. alpha=0 warning)
import warnings
warnings.filterwarnings("ignore")

# todo: write full notes section
USAGE = """
brutereg.py - Run a cross validated grid search of regression methods for an input set of predictors and descriptors

SYNOPSIS: 

  usage: brutereg.py -i <input_file> -o <output_file> [options]

OPTIONS:

  -h (--help)             display this message
  -i (--input=)           an input .csv file:- c1=reference c2=predictor c2-cn=descriptors
  -o (--output=)          an output pickle file containing the results of brutereg, which can be analysed with BruteSis
  -m (--min_train_score=) minimum R**2 score of training sets to keep models for (default = 0.75)
  -d (--max_diff=)        max difference between R**2 of training and test sets to keep models for (default = 0.15)
  -p (--train_percentage) the percentage of the input data to use for training (default = 50)
  -e (--estimators)       the estimators to use, as a list of numbers (e.g. [1,2,3] default = [1,2,3,4,5,6,7,8,9,10,11,12])
  -u (--hyperparameters)  input file for hyperparameters (default = './parameter_files/default_hyperparameter_grids')
  
      Estimator options:
      -----------------
      1: Random forest
      2: Extra random trees
      3: Simple OLS linear regression
      4: Ridge regression
      5: Ridge regression with cross validation (CV)
      6: Lasso (Least Absolute Shrinkage Selection Operator) regression
      7: Lasso with CV
      8: Lasso with least angle regression (lars) & CV
      9: Lasso lars with information criterion (IC) - AIC/BIC
      10: Elastic net regression
      11: Elastic net with CV
      12: Linear support vector regression

"""


def quality_filter(all_data, min_train_score, max_diff):

    results = all_data.results

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


def main(argv):
    # set
    input_file = ''
    output_file = ''
    min_train_score = 0.75
    max_diff = 0.15
    train_percentage = 50
    estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    input_params = './parameter_files/default_hyperparameter_grids'
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:d:p:e:u:",["help", "input", "output", "min_train_score", "max_diff",
                                                    "train_percentage", "estimators", "hyperparameters"])

    except getopt.GetoptError:
        print USAGE
        raise
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print USAGE
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg
        elif opt in ("-m", "--min_train_score"):
            min_train_score = float(arg)
        elif opt in ("-d", "--max_diff"):
            max_diff = float(arg)
        elif opt in ("-p", "--train_percentage"):
            train_percentage = float(arg)
        elif opt in ("-e", "--estimators"):
            estimators = eval(arg)
        elif opt in ("-u", "--hyperparameters"):
            input_params = arg


    if len(input_file) < 1 :
        print('ERROR: Must specify an input file!')
        sys.exit()

    if len(output_file) < 1:
        print('ERROR: Must specify an output file!')
        sys.exit()

    print USAGE
    print('The following estimator options were selected: ' + str(estimators) + '\n')

    print('**********************************************************************************\n')
    print('Running BruteReg on: ' + str(input_file))
    print('Will save project to: ' + str(output_file) + '\n')
    print('**********************************************************************************\n\n')

    print('Separating data from input for grid search...')
    X,y,labels = proj.set_input(str(input_file))
    print('Running grid search. Please note this will take a hell of a long time!')

    results = rg.auto_grid(X, y, labels, train_percentage, estimators, input_params)


    proj.save_eval(str(output_file), results)
    analysis_set = quality_filter(results, min_train_score, max_diff)
    proj.save_analysis(str(output_file), analysis_set)

if __name__ == "__main__":
    main(sys.argv[1:])
