# BruteReg - Brute-force Regression

Author: Rachael Skyner

Contact: rachael.skyner@diamond.ac.uk

Version: 0.1 (alpha)


### Info ###
This program was developed during my PhD, and was supported by funding from CCDC (https://www.ccdc.cam.ac.uk) 
and EPSRC (https://www.epsrc.ac.uk) under a St Andrews 600th anniversary scholarship.

It is designed to perform a brute-force cross-validated (3-fold) grid search over a number of different regression
estimators with pre-defined hyper-parameter grids. 

## Installation ##
Requirements: Python 2.7, pip and virtualenv

For the time-being, this program has only been tested on Linux and Mac. To install, either download the full
repository as a .zip file, or clone the repository with git. Next, run install_requirements.sh in your
terminal. 

Quick explanation of install:
1. A virtualenv of the program is created to isolate it from your machines python install - this is so that you
can install additional packages in a separate environment, which is useful if you don't have root access. 
2. Required packages are installed with pip in the virtualenv
3. PyInstaller (http://www.pyinstaller.org) is used to create a stand-alone exectuable for brutereg, located in:
   <path to download>/dist/brutereg/brutereg
   
   I suggest you add:
   
   alias brutereg='<path to download>/dist/brutereg/brutereg'
   
   to your bash profile in order to run the program with the command 'brutereg' from your terminal. From this
   point, we will refer to the executable as brutereg, assuming you have done this.

## Running BruteReg ##
This information can be displayed using brutereg -h:

brutereg - Run a cross validated grid search of regression methods for an input set of predictors and descriptors

SYNOPSIS: 

  usage: brutereg -i <input_file> -o <output_file> [-m <min_train_score> -d <max_train-test> -p <train_percentage>]

OPTIONS:

  -i (--input=)           an input .csv file:- c1=reference c2=predictor c2-cn=descriptors
  
  -o (--output=)          an output pickle file containing the results of brutereg, which can be analysed with BruteSis
  
  -m (--min_train_score=) minimum R**2 score of training sets to keep models for (default = 0.75)
  
  -d (--max_diff=)        max difference between R**2 of training and test sets to keep models for (default = 0.15)
  
  -p (--train_percentage=) the percentage of the input data to use for training (default = 50)
  
 ## Example files ## 
 There are two example input .csv files in 
