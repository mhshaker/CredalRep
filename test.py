import numpy as np
from scipy import stats

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([20,10,30,40,50,60,70,80,90])
tau, p_value = stats.kendalltau(x,y)
print(tau)