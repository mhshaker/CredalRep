
from scipy.stats import entropy
import numpy as np

# prob = [0.3, 0.7]
# prob = [0.2, 0.8]
prob = [0.4, 0.6]

res = entropy(prob)

prob = np.array(prob)
entropy = -prob*np.ma.log2(prob)
entropy = entropy.filled(0)
a = np.sum(entropy)

# print(res)
print(a)