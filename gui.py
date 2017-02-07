#!/usr/bin/python
# coding=utf8

import sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')

from tkinter import *
from tkinter import ttk
from tkFileDialog import *
import projecthandle as proj

from tkintertable import TableCanvas, TableModel 
from tkintertable.Tables_IO import TableImporter
import pandas as pd
import re
import pipemodules as pm

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import csv

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
         self.project = self.fileloader.load_file(self.fname)
         self.populate_tree()
      return self

   def populate_tree(self):
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
      #print len(dir(self.project))
      

   def _create_widgets(self): 
      self.main_frame()

   def main_frame(self):
      mainPanel = Frame(self,name='main')
      mainPanel.pack(side=TOP, fill=BOTH, expand=Y)

      nb = ttk.Notebook(mainPanel, name='notebook')
      nb.enable_traversal()
      nb.pack(fill=BOTH, expand=Y, padx=2, pady=3)

      ## Menu
      menubar = Menu(self.master)
   
      menu_file = Menu(menubar)
      menubar.add_cascade(menu=menu_file, label='Project')
      menu_file.add_command(label='Load Project', command=self.load_file)
      self.master['menu']=menubar
      ## Add tabs
      self.create_view_tab(nb)
      self.create_model_tab(nb)

   def reset_table(self):
      self.table = TableCanvas(self.tframe,rows=0,cols=0)
      self.table.createTableFrame()

   def create_view_tab(self,nb):
      self.frame = ttk.Frame(nb)
      self.frame.tree = ttk.Treeview(self.frame)
      self.frame.tree.grid(row=0, column=0, sticky=N+S+E+W)

      self.button_frame = Frame(self.frame)
      self.create_method_frame = Frame(self.frame)
      self.create_method_frame.grid(row=1,column=1,sticky=N+E)
      self.create_method_frame.label1=Label(self.create_method_frame, text='Index (from first column)')
      self.create_method_frame.label1.grid(row=0,column=0)
      self.create_method_frame.method_entry = Entry(self.create_method_frame)
      self.create_method_frame.method_entry.grid(row=0,column=1)
      self.create_method_frame.button=Button(self.create_method_frame, text="Set method", command=self.set_method)
      self.create_method_frame.button.grid(row=0,column=2)
            
      self.frame.button2 = Button(self.button_frame, text='Display', command=self.table_load, width=15)
      self.frame.button2.grid(row=0, column=0, sticky=N+E+W+S)
      
      self.frame.button3=Button(self.button_frame,text='Reset Display',command=self.reset_table, width=15)
      self.frame.button3.grid(row=1, column=0, sticky=N+E+W+S)

      self.button_frame.grid(row=1,column=0, sticky=N+E)

      self.frame.rowconfigure(0, weight=1)
      self.frame.columnconfigure((0,1), weight=2)
      
      self.tframe = Frame(self.frame)
      self.tframe.grid(row=0, column=1, sticky=N+S+E+W)
      self.table = TableCanvas(self.tframe,rows=0,cols=0)
      self.table.createTableFrame()

      self.f = Figure(figsize=(3,1), dpi=200)
      self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
      self.canvas.get_tk_widget().grid(row=3,column=1,sticky=N+S+E+W,ipady=50)

      #self.canvas._tkcanvas.grid

      nb.add(self.frame, text='View Project', underline=0, padding=2)

      return self
   
   def create_model_tab(self,nb):
      self.frame2 = Frame()
      nb.add(self.frame2, text='Create Model', underline=0, padding=2)
      self.frame2.button1=Button(self.frame2,text='yolo')
      self.frame2.button1.pack()
      return self

   def table_load(self):
      curItem = self.frame.tree.focus()
      print curItem
      
      eval('self.project.'+curItem).to_csv('./temp.csv')
      model=self.table.model
      importer = TableImporter()
      #importer.open_File('./temp.csv')
      print self.table.model
      dictionary = importer.ImportTableModel('./temp.csv')
      os.system('rm temp.csv')

      model.importDict(dictionary)
      self.table.redrawTable()

   def set_method(self):
      ind = [int(self.create_method_frame.method_entry.get())]
      
      curItem = self.frame.tree.focus()
      results = eval('self.project.'+curItem)
      for i in ind:     
         string = results.method_ids[i] # retrive method id
         setup = eval(string) # convert to iterable array

         temp = pm.search_random_forest() #initiate class

         # set the estimator type and initiate estimator class
         _,self.clf,_ = temp.set_method(setup[2]) 

         # get the development set features
         X_dev_temp, _ = pm.get_X(self.project.dev_set.matrix_raw, \
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

         # fit the estimator to the development set
         self.clf.fit(X_dev_temp, self.project.dev_set.y_raw)
         # predict the evaluation set
         eval_predict = self.clf.predict(X_eval_temp)
         # predict the development set - for metrics
         dev_predict = self.clf.predict(X_dev_temp)

      self.f.clear()
      self.f = Figure(figsize=(2,1), dpi=100)
      
      self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
      self.canvas.get_tk_widget().grid(row=3,column=1,sticky=N+S+E+W,ipady=50)
      self.a = self.f.add_subplot(111)
      for label in (self.a.get_xticklabels() + self.a.get_yticklabels()):
         label.set_fontsize(7)

      self.a.scatter(dev_predict,self.project.dev_set.y_raw, alpha=0.4, label="Development set")
      self.a.scatter(eval_predict,self.project.eval_set.y_raw, color='red', alpha=0.4, label="Evaluation set")
      self.a.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=1,  borderaxespad=0., fontsize=7)
      
      self.canvas.draw()
      

      print eval_predict
      return self.clf


if __name__ == "__main__":
    NotebookDemo().mainloop()
