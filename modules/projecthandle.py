#!/usr/bin/python
# coding=utf8

import pickle
import os.path
import projecthandle as proj

def save_eval(filename, all_data):
    devobj = proj.input_object()
    evalobj = proj.input_object()
    methobj = proj.method_object()
    projectobj = proj.project()
    devobj.create_object(all_data.X_train, all_data.Y_train, all_data.descriptor_matrix_train)
    evalobj.create_object(all_data.X_test, all_data.Y_test, all_data.descriptor_matrix_test)
    methobj.create_object(all_data.k_vals, all_data.selection_labels, all_data.ind_values, all_data.options)
    projectobj.save_project(all_data.results, devobj, evalobj, methobj, filename)
    
def save_analysis(analysis_results, filename):
    check = file_loader()
    verify = check.check_contents(filename)
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
                    if count == 5:
                        print 'This file contains an evaluation and analysis set'
                except:
                    val = False       
        f.close()
        return count

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
                print 'hi'
                return f
            if load_number == 4:
                f = eval_load()
                f.close()
            if load_number == 5:
                print 'hi'
                f = eval_load()
                self.analysis_results = pickle.load(f)
                f.close()
        except:
            raise IOError("Can't load file... check filetype and contents...")


class project(object):
    
    def __init__(self):
        return None
    
    def save_project(self, results, data_object, data_object2, method_object, filename):
        
        check = os.path.exists(filename)
        
        def save():
            with open(filename, 'wb') as pickle_place:
                pickle.dump(results, pickle_place)
                pickle.dump(data_object, pickle_place)
                pickle.dump(data_object2, pickle_place) 
                pickle.dump(method_object, pickle_place)
                print 'saving...'

                
        while check == False:
            save()
            check = os.path.exists(filename)
        else:
            user_var = raw_input('File exists... overwrite? [y/n]')
            if user_var == 'y' or user_var == 'n':
                if user_var == 'y':
                    save()
                if user_var == 'n':
                    new_filename = raw_input('Enter new filename: ')
                    filename = new_filename
                    save()
            else:
                user_var = raw_input('File exists... overwrite? [y/n]')
        
        return self


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
