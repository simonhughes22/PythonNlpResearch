#!/usr/bin/env bash
# A script to install all of the core dependencies to do research (minus deep learning libraries and other experimental items)
pip install -U pip
pip install pandas
pip install scikit-learn
pip install gensim
pip install nltk
pip install spacy
pip install pymongo
pip install jupyter
pip install ipython
# all above should have forced numpy and scipy to be installed
