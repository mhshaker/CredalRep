
from sklearn import datasets
import numpy as np



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

df = unpickle("./Data/Jdata/data_batch_1")
x = df.get("data")
print(x.shape)
# print(df)