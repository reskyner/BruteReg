#!/usr/bin/python
# coding=utf8

import sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')
try:
    from tkinter import *
    from tkinter import ttk
    from tkFileDialog import *
except:
    from Tkinter import *
    import ttk
    from tkFileDialog import *

import projecthandle as proj

from tkintertable import TableCanvas, TableModel 
from tkintertable.Tables_IO import TableImporter
import pandas as pd
import re
import pipemodules as pm
import run_grid as rg

from sklearn import metrics

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import csv


root = Tk()

# a fix for running on OSX - to center the title text vertically
if root.tk.call('tk', 'windowingsystem') == 'aqua':  # only for OSX
    s = ttk.Style()
    # Note: the name is specially for the text in the widgets
    s.configure('TNotebook.Tab', padding=(12, 8, 12, 0))

def quality_filter(results, min_train_score=0.75, max_diff=0.15):

    results.reset_index(drop=True, inplace=True)

    ## filter results and eliminate poor models
    for i in range(0,len(results)):
        if results.mean_train_score[i] > float(min_train_score) \
        and abs(results.mean_test_score[i] - results.mean_train_score[i]) < float(max_diff):
            continue
        else: 
            results.drop(i, axis=0, inplace=True)

    results.reset_index(drop=True, inplace=True)
    return results

def fit_method(self,ind,results):
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

   for i in ind:     
      string = results.method_ids[i] # retrive method id
      setup = eval(string) # convert to iterable array

      temp = pm.search_random_forest() #initiate class

      # set the estimator type and initiate estimator class
      _,self.clf,_ = temp.set_method(setup[2]) 

      # get the development set features
      X_dev_temp, self.descriptors = pm.get_X(self.project.dev_set.matrix_raw, \
                              self.project.meth.indvals[setup[0]][setup[1]]) 
      # get the evaluation set features
      X_eval_temp, _ = pm.get_X(self.project.eval_set.matrix_raw, \
                               self.project.meth.indvals[setup[0]][setup[1]]) 

      del temp
      
      # retreive hyper-parameters
      try:
         params = results['params'][i]
      except:
         params = results['parameters'][i]
      # set estimator hyper-parameters
      self.clf.set_params(**params)
      
      self.output_frame.output.insert(END,str(str(self.clf)+'\n\n'))

      # fit the estimator to the development set
      self.clf.fit(X_dev_temp, self.project.dev_set.y_raw)
      # predict the evaluation set
      eval_predict=[]
      eval_predict = self.clf.predict(X_eval_temp)
      # predict the development set - for metrics
      dev_predict=[]
      dev_predict = self.clf.predict(X_dev_temp)

      dev_set_score.append(self.clf.score(X_dev_temp, self.project.dev_set.y_raw))
      eval_set_score.append(self.clf.score(X_eval_temp, self.project.eval_set.y_raw))
      dev_evs.append(metrics.explained_variance_score(dev_predict, self.project.dev_set.y_raw))
      eval_evs.append(metrics.explained_variance_score(eval_predict, self.project.eval_set.y_raw))
      dev_mae.append(metrics.mean_absolute_error(dev_predict, self.project.dev_set.y_raw))
      eval_mae.append(metrics.mean_absolute_error(eval_predict, self.project.eval_set.y_raw))
      dev_mse.append(metrics.mean_squared_error(dev_predict, self.project.dev_set.y_raw))
      eval_mse.append(metrics.mean_squared_error(eval_predict, self.project.eval_set.y_raw))
      dev_medae.append(metrics.median_absolute_error(dev_predict, self.project.dev_set.y_raw))
      eval_medae.append(metrics.median_absolute_error(eval_predict, self.project.eval_set.y_raw))
      method_ids.append(string)
      parameters.append(params)

      if hasattr(self.clf,'coef_'):

          string2="equation = "

	  string2 += str(self.clf.intercept_)

          for i in range(0,len(self.clf.coef_)):
              if self.clf.coef_[i] !=0:
                  string2 += str(' + ' + str(self.clf.coef_[i]) + '*' + str(self.descriptors[i] + '\n'))

          string2 += '\n'

          
          self.output_frame.output.insert(END,string2)

   # create dictionary object from results
   evaluation_results = {'dev_set_score':dev_set_score, 'eval_set_score':eval_set_score, \
                      'method_ids':method_ids, 'parameters':parameters, 'dev_evs':dev_evs, \
                      'eval_evs':eval_evs, 'dev_mae':dev_mae, 'eval_mae':eval_mae, \
                      'dev_mse': dev_mse, 'eval_mse':eval_mse, 'dev_median_ae':dev_medae, \
                      'eval_median_ae':eval_medae}
   evaluation_results=pd.DataFrame.from_dict(evaluation_results)

   return eval_predict, dev_predict, evaluation_results

class NotebookDemo(ttk.Frame):
   
   def __init__(self, isapp=True, name='notebookdemo'):
      ttk.Frame.__init__(self, name=name)
      self.pack(expand=Y, fill=BOTH)
      self.master.title('BruteSis v0.1(d)')
      self.isapp = isapp
      self._create_widgets()
      

   def load_file(self):
      
      self.fname = askopenfilename()
      if self.fname:
         self.fileloader = proj.file_loader()
         self.project, number = self.fileloader.load_file(self.fname)
         self.populate_tree()
         if number<5:
            self.toplevel=Toplevel()

            message = 'This project only contains raw grid search results.\n\
Create an analysis set by filtering out poor results.\n\n\
min_train score: minimum average score of the training set\n\
max_diff: maximum difference between mean train and test scores\n'
                       

            self.toplevel.message = Label(self.toplevel, text=message, wraplength=3000)
            self.toplevel.message.grid(row=0)
            self.toplevel.second_frame=ttk.Frame(self.toplevel)
            self.toplevel.second_frame.grid(row=1)

            self.toplevel.button2=Button(self.toplevel.second_frame, text="Quality Filter", command=self.qfilter_button)
            self.toplevel.button2.grid(row=1,column=4)
            self.toplevel.mintscore = Entry(self.toplevel.second_frame,width=5)
            self.toplevel.mintscore.grid(row=1,column=1)
            self.toplevel.label2=Label(self.toplevel.second_frame, text='min_train_score')
            self.toplevel.label2.grid(row=1,column=0)
            self.toplevel.mindiff = Entry(self.toplevel.second_frame, width=5)
            self.toplevel.mindiff.grid(row=1,column=3)
            self.toplevel.label3=Label(self.toplevel.second_frame, text='max_diff')
            self.toplevel.label3.grid(row=1,column=2)

      
             
      
   def populate_tree(self):
      self.menu_file.entryconfig('Open Project', state='disabled')
      underscores=re.compile('\__')
      count = 0
      parentNode=""
      results = self.frame.tree.insert('','end','',text='Results')
      typedf = str("<class 'pandas.core.frame.DataFrame'>")
      
      for i in dir(self.project):
         m = re.match(underscores, i)
         
         if m<=0:
            parent_type = eval('str(self.project.'+i+'.__class__)')
         
            if parent_type==typedf:
               self.frame.tree.insert(results, 'end', str(i), text=str(i))
            
         count +=1
      #print count
      #print len(dir(project))     

   def _create_widgets(self): 
      self.main_frame()

   def new_project(self):

      self.frame2 = self.create_model_tab()
      self.nb.add(self.frame2, text="Create Project")
      
      self.nb.tab(1,state='normal')
      self.nb.select(1)
      self.menu_file.entryconfig("New Project", state='disabled')

      

   def main_frame(self):
      self.mainPanel = ttk.Frame(name='main')
      self.mainPanel.pack(side=TOP, fill=BOTH, expand=Y)

      self.nb = ttk.Notebook(self.mainPanel, name='notebook')

      ## Menu
      menubar = Menu(self.master)

      self.menu_file = Menu(menubar)
      
      menubar.add_cascade(menu=self.menu_file, label='Project')
      
      self.menu_file.add_command(label='Open Project', command=self.load_file)
      self.menu_file.add_command(label='Clear Project', command=self.__init__)
      self.menu_file.add_command(label='New Project', command=self.new_project)
   
      self.master['menu']=menubar

      ## Add tabs
      self.nb.pack(fill=BOTH, expand=Y, padx=2, pady=3)

      self.create_view_tab()
      self.nb.add(self.frame, text="View Project")
      self.nb.select(0)
      

   def reset_table(self):
      self.table = TableCanvas(self.tframe,rows=0,cols=0)
      self.table.createTableFrame()
      self.frame.button3.config(state='disabled')
      self.frame.button2.config(state='normal')

   def create_view_tab(self):
      def textclear():
          self.output_frame.output.delete(1.0,END)
          
      self.frame = ttk.Frame(self.nb)
      self.frame.grid(row=0,column=0)
      self.frame.tree = ttk.Treeview(self.frame)
      self.frame.tree.grid(row=0, column=0, sticky=N+S+E+W)

      self.output_frame = ttk.Frame(self.frame)
      self.output_frame.grid(row=3,column=0,sticky=N+S+E)
      self.output_frame.scroll=ttk.Scrollbar(self.output_frame)
      self.output_frame.scroll2=ttk.Scrollbar(self.output_frame, orient=HORIZONTAL)
      self.output_frame.output=Text(self.output_frame,width=50,wrap=NONE)
      self.output_frame.scroll.grid(row=0,column=1,sticky=N+S+E+W)
      self.output_frame.scroll2.grid(row=1,column=0,sticky=S+E+W)
      self.output_frame.output.grid(row=0,column=0,sticky=N+W)
      self.output_frame.output.config(yscrollcommand=self.output_frame.scroll.set)
      self.output_frame.output.config(xscrollcommand=self.output_frame.scroll2.set)
      self.output_frame.scroll.config(command=self.output_frame.output.yview)
      self.output_frame.scroll2.config(command=self.output_frame.output.xview)
      
      self.button_frame = ttk.Frame(self.frame)
      self.create_method_frame = ttk.Frame(self.frame)
      self.create_method_frame.grid(row=1,column=1,sticky=N+E)
      self.create_method_frame.label1=ttk.Label(self.create_method_frame, text='Index')
      self.create_method_frame.label1.grid(row=0,column=2, sticky=N+E)
      self.create_method_frame.method_entry = ttk.Entry(self.create_method_frame)
      self.create_method_frame.method_entry.grid(row=0,column=3, sticky=N+E)
      self.create_method_frame.button= ttk.Button(self.create_method_frame, text="Plot method", command=self.set_method)
      self.create_method_frame.button.grid(row=0,column=4, sticky=N+E)
      self.create_method_frame.button2= ttk.Button(self.create_method_frame, text="Filter", command=self.filters_pane)
      self.create_method_frame.button2.grid(row=0,column=5, sticky = N+E)
      
      self.frame.button2 = ttk.Button(self.button_frame, text='Display', command=self.table_load, width=15)
      self.frame.button2.grid(row=0, column=0, sticky=N+E+W+S)
      
      self.frame.button3= ttk.Button(self.button_frame,text='Reset Display',command=self.reset_table, width=15)
      self.frame.button3.grid(row=0, column=1, sticky=N+E+W+S)

      self.frame.button4 = ttk.Button(self.frame, text='Clear', command=textclear)
      self.frame.button4.grid(row=4,column=0,sticky=N+E)

      self.button_frame.grid(row=1,column=0, sticky=N+E)
      
      self.tframe = ttk.Frame(self.frame)
      self.tframe.grid(row=0, column=1, sticky=N+S+E+W)
      
      self.table = TableCanvas(self.tframe,rows=0,cols=0)
      self.table.createTableFrame()

      self.f = Figure(figsize=(3,1), dpi=200)
      self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
      self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame).grid(row=4, column=1, sticky=S+E+W)
      self.canvas.get_tk_widget().grid(row=3,column=1,sticky=N+S+E+W,ipady=50)
      self.canvas.draw()

      self.frame.rowconfigure(0, weight=1)
      self.frame.columnconfigure(1, weight=2)


   def table_load(self):
      curItem = self.frame.tree.focus()
      print curItem

      eval('self.project.'+curItem).to_csv('./temp.csv')
      model=self.table.model
      importer = TableImporter()
      #importer.open_File('./temp.csv')
      print self.table.model
      dictionary = importer.ImportTableModel('./temp.csv')
      #os.system('rm temp.csv')

      model.importDict(dictionary)
      self.table.redrawTable()
      self.frame.button2.config(state='disabled')
      self.frame.button3.config(state='normal')

   def descriptors_filter(self):
       self.descriptors_frame=ttk.Frame(self.filter_pane)
       self.descriptors_frame.grid(row=1,column=0)
       self.descriptors_frame.l1=ttk.Label(self.descriptors_frame, text='Min no.')
       self.descriptors_frame.l1.grid(row=0,column=0)
       min_descriptors = self.descriptors_frame.o1=ttk.Combobox(self.descriptors_frame)
       self.descriptors_frame.o1.grid(row=0,column=1)
       self.descriptors_frame.l2=ttk.Label(self.descriptors_frame, text='Max no.')
       self.descriptors_frame.l2.grid(row=1,column=0)
       max_descriptors = self.descriptors_frame.o2=ttk.Combobox(self.descriptors_frame)
       self.descriptors_frame.o2.grid(row=1,column=1)

       curItem = self.frame.tree.focus()
       results = eval('self.project.'+curItem)
       ind=range(0,len(results))
       descriptor_numbers=[]
       for i in ind:
           fit_method(self,[i],results)
           count=0

           if hasattr(self.clf,'coef_'):
               for i in self.clf.coef_:
                   if abs(i)>0:
                       count+=1
               descriptor_numbers.append(count)
           if hasattr(self.clf,'feature_importances_'):
               for i in self.clf.feature_importances_:
                   if abs(i)>0:
                       count+=1
               descriptor_numbers.append(count)

       min_vals = range(min(descriptor_numbers),max(descriptor_numbers)-1)
       max_vals = range(min(descriptor_numbers),max(descriptor_numbers))

       max_descriptors['values']=max_vals
       min_descriptors['values']=min_vals

       self.descriptor_numbers=descriptor_numbers

       self.descriptors_frame.button=ttk.Button(self.descriptors_frame, text='Run', command=self.new_table_descriptors)
       self.descriptors_frame.button.grid(row=2,column=1, sticky=E)

       
       self.project.new_table=pd.DataFrame(results)

   def new_table_descriptors(self):

        droplist = []
        for i in range(0,len(self.descriptor_numbers)):
            if float(self.descriptor_numbers[i]) < float(self.descriptors_frame.o1.get()) or self.descriptor_numbers[i] > float(self.descriptors_frame.o2.get()):
                   droplist.append(i)

        self.project.new_table.drop(droplist, axis=0, inplace=True)

        self.reset_table()

        print droplist

        self.descriptor_filt = self.frame.tree.insert('','end','',text='Descriptors Filter')
        self.frame.tree.insert(self.descriptor_filt, 'end', 'new_table', text='new_table')
        
   def filters_pane(self):

           
       self.create_method_frame.button2.config(state='disabled')
       
       self.filter_pane=ttk.Frame(self.frame)
       self.filter_pane.grid(row=0,column=2,sticky=N+S+E, padx=10, pady=10)

       self.filter_pane.p1 = ttk.Labelframe(master=self.filter_pane, text='Filters')
       self.filter_pane.p1.grid(row=0,column=0,sticky=N+S+E)
       self.filter_pane.button1=ttk.Button(master=self.filter_pane.p1, text='Descriptors', command=self.descriptors_filter)
       self.filter_pane.button1.grid(row=0,column=1)
       self.filter_pane.button2=ttk.Button(master=self.filter_pane.p1, text='Error')
       self.filter_pane.button2.grid(row=0,column=2)
       self.filter_pane.button3=ttk.Button(master=self.filter_pane.p1, text='Variance')
       self.filter_pane.button3.grid(row=0,column=3)


       

          
   def create_model_tab(self):
      
      self.frame2 = ttk.Frame(self.nb)
      self.frame2.pack()

      self.frame2.methods=ttk.Labelframe(master=self.frame2, text='Methods')
      self.frame2.methods.grid(row=0,column=0,padx=20, pady=10, sticky=N+S+E+W)

      self.frame2.methods.check1 = ttk.Checkbutton(master=self.frame2.methods, text="Random Forest", state='disabled')
      self.frame2.methods.check1.grid(row=0,column=0, sticky=N+S+W, padx=10)
      self.frame2.methods.check2 = ttk.Checkbutton(master=self.frame2.methods, text="Extra Trees", state='disabled')
      self.frame2.methods.check2.grid(row=1,column=0, sticky=N+S+W, padx=10)
      self.frame2.methods.check3 = ttk.Checkbutton(master=self.frame2.methods, text="Linear Regression", state='disabled')
      self.frame2.methods.check3.grid(row=2,column=0, sticky=N+S+W, padx=10)
      self.frame2.methods.check4 = ttk.Checkbutton(master=self.frame2.methods, text="Ridge", state='disabled')
      self.frame2.methods.check4.grid(row=3,column=0, sticky=N+S+W, padx=10)
      self.frame2.methods.check5 = ttk.Checkbutton(master=self.frame2.methods, text="Ridge CV", state='disabled') ## DON'T ENABLE
      self.frame2.methods.check5.grid(row=4,column=0, sticky=N+S+W, padx=10)
      self.frame2.methods.check6 = ttk.Checkbutton(master=self.frame2.methods, text="Lasso", state='disabled')
      self.frame2.methods.check6.grid(row=5,column=0, sticky=N+S+W, padx=10)
      self.frame2.methods.check7 = ttk.Checkbutton(master=self.frame2.methods, text="Lasso CV", state='disabled')
      self.frame2.methods.check7.grid(row=0,column=1, sticky=N+S+W, padx=10)
      self.frame2.methods.check8 = ttk.Checkbutton(master=self.frame2.methods, text="Lasso Lars CV", state='disabled')
      self.frame2.methods.check8.grid(row=1,column=1, sticky=N+S+W, padx=10)
      self.frame2.methods.check9 = ttk.Checkbutton(master=self.frame2.methods, text="Lasso Lars IC", state='disabled')
      self.frame2.methods.check9.grid(row=2,column=1, sticky=N+S+W, padx=10)
      self.frame2.methods.check10 = ttk.Checkbutton(master=self.frame2.methods, text="Elastic Net", state='disabled')
      self.frame2.methods.check10.grid(row=3,column=1, sticky=N+S+W, padx=10)
      self.frame2.methods.check11 = ttk.Checkbutton(master=self.frame2.methods, text="Elastic Net CV", state='disabled')
      self.frame2.methods.check11.grid(row=4,column=1, sticky=N+S+W, padx=10)
      self.frame2.methods.check12 = ttk.Checkbutton(master=self.frame2.methods, text="Linear SVR", state='disabled')
      self.frame2.methods.check12.grid(row=5,column=1, sticky=N+S+W, padx=10)

      self.frame2.kvals=ttk.Labelframe(master=self.frame2, text='K-values')
      self.frame2.kvals.grid(row=0,column=1, sticky=N+S+E+W, padx=20, pady=10)

      self.frame2.kvals.label1 = ttk.Label(master=self.frame2.kvals, text='Min. no. of descriptors')
      self.frame2.kvals.label1.grid(row=0,column=0,sticky=N+S+W, padx=10)
      self.frame2.kvals.list1 = ttk.Combobox(self.frame2.kvals, state='disabled')
      self.frame2.kvals.list1.grid(row=0,column=1)
      self.frame2.kvals.label2 = ttk.Label(master=self.frame2.kvals, text='Max. no. of descriptors')
      self.frame2.kvals.label2.grid(row=1,column=0,sticky=N+S+W, padx=10)
      self.frame2.kvals.list2 = ttk.Combobox(self.frame2.kvals, state='disabled')
      self.frame2.kvals.list2.grid(row=1,column=1)
      self.frame2.kvals.label3 = ttk.Label(master=self.frame2.kvals, text='Increment')
      self.frame2.kvals.label3.grid(row=2,column=0,sticky=N+S+W, padx=10)
      self.frame2.kvals.list3 = ttk.Combobox(self.frame2.kvals, state='disabled')
      self.frame2.kvals.list3.grid(row=2,column=1)

      self.nb.pack()

      return self.frame2


   def qfilter_button(self):
      
      min_train_score = self.toplevel.mintscore.get()
      max_diff = self.toplevel.mindiff.get()
      curItem = self.frame.tree.focus()
      results = eval('self.project.eval_results')
      filtered = quality_filter(results,min_train_score,max_diff)
      print filtered
      _,_,self.project.temp_results = fit_method(self,range(0,len(filtered)),filtered)

      #fil_tree = self.frame.tree.insert('','end','',text='Filter Results')
      #self.frame.tree.insert('results', 'end', 'temp_results', text='analysis_set')

      proj.save_analysis(self.project.temp_results,self.fname)
      self.__init__()
      self.toplevel.destroy()


   def set_method(self):
      ind = [int(self.create_method_frame.method_entry.get())]
      
      curItem = self.frame.tree.focus()
      results = eval('self.project.'+curItem)

      eval_predict,dev_predict,_=fit_method(self,ind,results)

      self.f.clear()
      self.f = Figure(figsize=(2,1), dpi=100)
      
      self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
      self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame).grid(row=4, column=1, sticky=S+E+W)
      self.canvas.get_tk_widget().grid(row=3,column=1,sticky=N+S+E+W,ipady=50)
      self.a = self.f.add_subplot(111)
      for label in (self.a.get_xticklabels() + self.a.get_yticklabels()):
         label.set_fontsize(7)

      self.a.scatter(dev_predict,self.project.dev_set.y_raw, alpha=0.4, label="Development set")
      self.a.scatter(eval_predict,self.project.eval_set.y_raw, color='red', alpha=0.4, label="Evaluation set")
      self.a.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=1,  borderaxespad=0., fontsize=7)
      
      self.canvas.draw()


if __name__ == "__main__":
    NotebookDemo().mainloop()
