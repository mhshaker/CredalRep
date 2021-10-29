from math import log2
import numpy as np
import random
from scipy.optimize import minimize

def constarint(x):
    sum_x = np.sum(x)
    return 1 - sum_x

def convex_ent_max30(p, l):
	l = np.reshape(l,(-1,1))
	l_p = l * p
	l_p_sum = np.sum(l_p, axis=0)
	l_sum = np.sum(l, axis=0)
	z = l_p_sum / l_sum # to get the formual for S^* in paper
	entropy = -z*np.log2(z)
	entropy_sum = np.sum(entropy)
	return entropy_sum * -1 # so that we maximize it

def convex_ent_min30(p, l):
	l = np.reshape(l,(-1,1))
	l_p = l * p
	l_p_sum = np.sum(l_p, axis=0)
	l_sum = np.sum(l, axis=0)
	z = l_p_sum / l_sum # to get the formual for S^* in paper
	entropy = -z*np.log2(z)
	entropy_sum = np.sum(entropy)
	return entropy_sum  # so that we maximize it

data_point_prob = np.array([[0.1,0.9], [0.9,0.1]])
likelyhoods = np.ones((data_point_prob.shape[0]))
likelyhoods_sum = np.sum(likelyhoods)
likelyhoods = likelyhoods / likelyhoods_sum


cons = ({'type': 'eq', 'fun': constarint})


x0_index = 1
x0 = data_point_prob[x0_index]
bnds = []
for class_index in range(data_point_prob.shape[1]):
    b_min = data_point_prob[:,class_index].min()
    b_max = data_point_prob[:,class_index].max()
    bnds.append((b_min, b_max))
opt_res = minimize(convex_ent_max30, x0, args=(likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
opt_entropy = -opt_res.fun
opt_prob = opt_res.x
print("------------------------------------opt max")
print(f"opt_entropy {opt_entropy}")
print(f"opt_prob {opt_prob}")

opt_res = minimize(convex_ent_min30, x0, args=(likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
opt_entropy = opt_res.fun
opt_prob = opt_res.x
print("------------------------------------opt min")
print(f"opt_entropy {opt_entropy}")
print(f"opt_prob {opt_prob}")


print("------------------------------------")
h = (0.1) * log2(0.1) + (0.9) * log2(0.9)

print(h)