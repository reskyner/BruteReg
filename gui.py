import sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')

from Tkinter import *
from tkFileDialog import *
import projecthandle as proj
import run_grid as rg

from tkintertable import TableCanvas, TableModel 
from tkintertable.Tables_IO import TableImporter
import pandas as pd

import csv


#from Tkinter.messagebox import showerror

def donothing():
   print 'Ha!'

class MyFrame(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("BruteReg v0.1(DevEdn.)")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W+E+N+S)

        self.button2 = Button(self, text='Display', command=self.table_load, width=15)
        self.button2.grid(row=1, column=1, sticky=W)

        self.tframe = Frame()
        self.tframe.grid(row=3, column=0, sticky=W)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        fl = Menu(menu)
        fl.add_command(label="Load input", command=self.load_file)
        menu.add_cascade(label="File", menu=fl)
        
        calculation = Menu(menu)
        calculation.add_command(label="Run default grid", command=self.poo)
        menu.add_cascade(label="Calculation", menu=calculation)
        

    def table_load(self):
        #for i in self.X:
            #self.X[i] = str(i)
        self.dic = self.X.to_csv('.temp.csv')
        importer = TableImporter()
        self.model = TableModel()
        self.dictionary = importer.ImportTableModel('.temp.csv')
        os.system("rm .temp.csv")
        self.model.importDict(self.dictionary)
        self.table = TableCanvas(self.tframe, model=self.model)

        self.table.createTableFrame()
        self.table.redrawTable()
        #

        print 'why me no work...?'
        return self
        
        return
    def load_file(self):
        self.fname = askopenfilename()
        if self.fname:
            #try:
                self.X, self.y ,self.labels = proj.set_input(self.fname)
                return self
                #tframe = Frame(MyFrame)
                #tframe.pack()
                #table = TableCanvas(tframe)
                #table.createTableFrame()
            #except:                     # <- naked except is a bad idea
                #showerror("Uh-Oh...", "Failed to read file\n'%s'" % fname)
        return

    def poo(self):
        self.results = rg.auto_grid(self.X, self.y, labels)
        return


if __name__ == "__main__":
    MyFrame().mainloop()
