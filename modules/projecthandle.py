import pickle
import pandas as pd


class input_object(object):
    def __init__(self):
        return None

    def create_object(self, X, y, matrix):
        self.X_raw = X
        self.y_raw = y
        self.matrix_raw = matrix


class method_object(object):
    def __init__(self):
        return None

    def create_object(self, k, selection, indicies, methods):
        self.kvals = k
        self.methodmatrix = selection
        self.indvals = indicies
        self.methodopts = methods


class file_loader(object):
    def __init__(self):
        None

    def load_file(self, filename):
        try:
            load_number = check_contents(filename)
            print load_number

            def eval_load():
                # import evaluation set
                f = open(filename, 'rb')
                self.eval_results = pickle.load(f)
                self.dev_set = pickle.load(f)
                self.eval_set = pickle.load(f)
                self.meth = pickle.load(f)

                return f

            if load_number == 4:
                f = eval_load()
                f.close()
            if load_number == 5:
                f = eval_load()
                self.analysis_results = pickle.load(f)
                f.close()
        except:
            raise IOError("Can't load file... check filetype and contents...")
        return self, load_number


class project(object):
    def __init__(self):
        return None

    def save_project(self, results, data_object, data_object2, method_object, filename):

        print('Attempting to save to: ' + str(filename))
        def save():
            with open(filename, 'wb') as pickle_place:
                pickle.dump(results, pickle_place)
                pickle.dump(data_object, pickle_place)
                pickle.dump(data_object2, pickle_place)
                pickle.dump(method_object, pickle_place)
                print 'saving...'

        save()

        return self

def set_input(filename):
    """set_input('filename') 

    Import a csv file of the format:
    1. first row - should be names of labels, property, then n descriptors:
       labels, y, X1-name....Xn-name

    2. following rows - values corresponding to column headers:
       label1, y1, X1.....Xn"""

    descriptors_raw = pd.read_csv(filename)
    labels = descriptors_raw.iloc[:, 0]
    X = descriptors_raw.iloc[:, 2:]
    y = descriptors_raw.iloc[:, 1]

    del descriptors_raw
    return X, y, labels

def save_eval(filename, all_data):
    proj=project()
    devobj = input_object()
    evalobj = input_object()
    methobj = method_object()
    projectobj = project()
    devobj.create_object(all_data.X_train, all_data.Y_train, all_data.descriptor_matrix_train)
    evalobj.create_object(all_data.X_test, all_data.Y_test, all_data.descriptor_matrix_test)
    methobj.create_object(all_data.k_vals, all_data.selection_labels, all_data.ind_values, all_data.options)
    projectobj.save_project(all_data.results, devobj, evalobj, methobj, filename)
    
def save_analysis(filename, analysis_results):
    check = file_loader()
    verify = check_contents(filename)
    if verify == 4:
        with open(filename, 'ab') as f:
            pickle.dump(analysis_results, f)
            f.close()
    else:
        raise IOError('Trying to write to wrong type of file, or analysis set already saved! Check the file...')


def check_contents(filename):
        count = 0
        val = True
        with open(filename, 'rb') as f:
            while val == True:
                try:
                    pickle.load(f)
                    count += 1
                    #if count == 4:
                        #print 'This file contains only an evaluation set'
                    #if count == 5:
                        #print 'This file contains an evaluation and analysis set'
                except:
                    val = False       
        f.close()
        return count

