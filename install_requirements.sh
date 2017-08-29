virtualenv ../BruteReg
source bin/activate
pip install 'matplotlib>=1.3.1'
pip install 'numpy>=1.11.0'
pip install 'pandas>=0.18.1'
pip install 'tkintertable>=1.2'
pip install 'SciPy>=0.13.3'
pip install 'scikit_learn>=0.19.0'
pip install 'pyinstaller'
pyinstaller -p ./modules/ -p ./lib/python2.7/site-packages/ --hidden-import sklearn.neighbors.typedefs  --hidden-import sklearn.neighbors.quad_tree --hidden-import sklearn.tree._utils brutereg.py

