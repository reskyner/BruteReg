#!/usr/bin/python
# coding=utf8

import sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('./modules')

from Tkinter import *
from tkFileDialog import *
import projecthandle as proj


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
        

    #def poo(self):
        #self.results = rg.auto_grid(self.X, self.y, self.labels)
        #return


if __name__ == "__main__":
    MyFrame().mainloop()
