[![Build Status](https://travis-ci.org/reskyner/BruteReg.svg)](https://travis-ci.org/reskyner/BruteReg)
<a href="https://codeclimate.com/github/reskyner/BruteReg"><img src="https://codeclimate.com/github/reskyner/BruteReg/badges/gpa.svg" /></a>
<a href="https://codeclimate.com/github/reskyner/BruteReg/"><img src="https://codeclimate.com/github/reskyner/BruteReg/badges/issue_count.svg" /></a>

# BruteReg - Brute-force Regression (for cheminformatics)

Author: Rachael Skyner

Contact: rachael.skyner@diamond.ac.uk

Version: 0.1 (alpha)


### Info ###
This program was developed during my PhD, and was supported by funding from CCDC (https://www.ccdc.cam.ac.uk) and EPSRC (https://www.epsrc.ac.uk) under a St Andrews 600th anniversary scholarship.

It is designed to perform a brute-force cross-validated (3-fold) grid search over a number of different regression estimators with pre-defined hyper-parameter grids. The regression models used in this work are implemented from scikit learn. Documentation here: http://scikit-learn.org/stable/

The program will produce 10's of thousands of models, depending on the number of descriptors and estimators you decide to use, so will take a while. The code has been optimised to make full and automated use of your CPUs, so is best used in a HPC envronment.

## Installation ##
Requirements: Python 2.7, pip and virtualenv

If you don't have root access and need to install locally, use something like anaconda (https://www.anaconda.com) as a python interpreter. Make sure that you are using the anaconda python executable rather than the system default version.

For the time-being, this program has only been tested on Linux and Mac. To install, either download the full repository as a .zip file, or clone the repository with git. Next, run install_requirements.sh in your terminal. 

Reccomended: use git clone so that you can automatically update by running update.sh (WARNING: This will erase any files that are not present in the git repository, so do not use the install directory as a working directory if you plan on updating this way!)

Quick explanation of install:
1. A virtualenv of the program is created to isolate it from your machines python install - this is so that you can install additional packages in a separate environment, which is useful if you don't have root access. 
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

  usage: brutereg.py -i <input_file> -o <output_file> [options]
  

OPTIONS:

  -h (--help)             display this message
  
  -i (--input=)           an input .csv file:- c1=reference c2=predictor c2-cn=descriptors
  
  -o (--output=)          an output pickle file containing the results of brutereg, which can be analysed with BruteSis
  
  -m (--min_train_score=) minimum R squared score of training sets to keep models for (default = 0.75)
  
  -d (--max_diff=)        max difference between R squared of training and test sets to keep models for (default = 0.15)
  
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
  
  Parallelisation of jobs, where possible, is automated, so no extra options are required beyond your servers queue handling software. It is recommended that you run BruteReg on an HPC, as it is very slow on a local machine (more cores = more parallelisation)
  
 ## Example files ## 
 There are two example input .csv files in ./input_files. These are two datasets that I compiled suring my PhD, one for hydration free energies and one for solubility (values shown in the second column). The descriptors (columns two onwards) for each structure were calculated with rdkit from SMILES strings.I haven't included the SMILES strings for these datasets, as the structural information belongs to CCDC. The refcodes (column 1) refer to the refcodes used in the CSD (CCDC's database of small molecule crystal structures).

Your own input files should follow the same format. A reference to the structure in the first column (this could be a smiles string or compound name, for example), the experimental value to be predicted in the second column, and descriptors in the following columns, with each structure having one row.

I will add a script to calculate rdkit descriptors from SMILES strings in the future. I will also aim to add a number of scripts to scrape databases (e.g. ChEMBL) for experimental properties information, although this is currently not implemented.

## Additional scripts ##
1. A script to calculate rdkit descriptors for an input csv file, with the column headings Refcode and Smiles, containing the smiles strings for the compounds (identified by a refcode - which can be anything) you wish to work with. This script is located in ./calculate_descriptors. 

   To run:

   python descriptor_calc.py -i <input_file> -o <output_file>

## Coming soon... ##
1. A tested and deployable version of the sister analysis GUI (BruteSis)
2. Scripts to scrape databases for experimental data
3. Better deployment through a Docker image
4. Support for Windows
