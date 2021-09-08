
import numpy as np
import UncertaintyM as unc
import matplotlib.pyplot as plt


# fix values
likelyhoods = np.array([0.9,0.1])
# likelyhoods = np.array([1/3,1/3,1/3])
porb_matrix = np.array([[0,1], [1,0]]) # 2 models and 2 classes
# porb_matrix = np.array([[1/3,2/3], [2/3,1/3]]) 
epsilon = 2

# call set18

def convex_ent_max18(s, p, l):
	s = np.reshape(s,(-1,1))
	l = np.reshape(l,(-1,1))
	s_l_p = s * l * p
	s_l = s * l
	s_l_p_sum = np.sum(s_l_p, axis=0)
	s_l_sum = np.sum(s_l, axis=0)
	z = s_l_p_sum / s_l_sum # to get the formual for S^* in paper
	# entropy = -z*np.log2(z)
	# entropy_sum = np.sum(entropy)
	return z # entropy_sum * -1 # so that we maximize it

m  = len(likelyhoods)
_m = 1/m

cons = ({'type': 'eq', 'fun': unc.constarint})
b = (_m * (1 / epsilon), _m * epsilon) # (_m - epsilon, _m + epsilon) addetive constraint
bnds = [ b for _ in range(m) ]

# sample multiple times
credal_set = []
for x in range(1000):
    s_r = unc.get_random_with_constraint(porb_matrix.shape[1],bnds)
    q = convex_ent_max18(s_r, porb_matrix,likelyhoods)
    credal_set.append(q)
    # print(q)

# plot
credal_set = np.array(credal_set)
credal_set_class0 = credal_set[:,0]
credal_set_class1 = credal_set[:,1]
# credal_set_class1 = credal_set[:,2]

# val = 0. # this is the value where you want the data to appear on the y-axis.
for c in range(porb_matrix.shape[1]):
	plt.plot(credal_set[:,c], np.zeros_like(credal_set_class0) + c, 'x')

# plt.plot(credal_set)
plt.savefig(f"./pic/unc/convex.png",bbox_inches='tight')
plt.close()