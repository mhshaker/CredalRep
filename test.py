
from sklearn import datasets
import numpy as np


f, t = datasets.load_svmlight_file("Data/Jdata/dbpedia_train.svm")

print(f.shape)
print(t.shape)