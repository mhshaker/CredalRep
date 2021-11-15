
from sklearn import datasets
import numpy as np


features, target = datasets.load_svmlight_file("Data/Jdata/dbpedia_train.svm")
print(features.shape)
print(target.shape)
print(np.unique(target))