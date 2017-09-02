import sys
import pandas as pd
sys.path.append('/usr/local/lib/python2.7/site-packages')
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import rdkit.Chem as Chem
import getopt

USAGE = """
descriptor_calc.py - Calculate rdkit descriptors for an input CSV file containing a Refcode and Smiles column

e.g.:

REFCODE, Smiles
mol1, CCCC
mol2, CC
mol3, CCCCC

USAGE:

    -i      input .csv file
    -o      output .csv file with descriptors

"""


def main(argv):
    input_file = ''
    output_file = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["help", "input", "output"])

    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg

    test = pd.read_csv(input_file)

    temp2 = [' ' for i in range(0, len(test))]

    nms = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    for i in range(0, len(nms)):
        test.insert(test.shape[1],nms[i],temp2)

    for i in range(0,len(test["Smiles"])):
        m = Chem.MolFromSmiles(test["Smiles"].iloc[i])
        descr = calc.CalcDescriptors(m)
        for x in range(len(descr)):
            name = str(nms[x])
            test[name].iloc[i] = str(descr[x])

    pd.DataFrame.to_csv(test, output_file)

if __name__ == "__main__":
    main(sys.argv[1:])
