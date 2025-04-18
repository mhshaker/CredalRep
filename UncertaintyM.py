import math
import numpy as np
from numpy.core.fromnumeric import ptp
import pandas as pd
import matplotlib.pyplot as plt
import random
import time 
import itertools
from scipy.optimize import minimize
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn import metrics
from collections import Counter
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog
from scipy.stats import entropy
from scipy import stats
from statistics import mean
import warnings
random.seed(1)

################################################################################################################################################# ent

def sk_ent(probs):
	uncertainties = np.array([entropy(prob) for prob in probs])
	return uncertainties, uncertainties, uncertainties # all three are the same as we only calculate total uncertainty

def uncertainty_ent(probs): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = np.array(probs)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	a = np.sum(entropy, axis=1)
	a = np.sum(a, axis=1) / entropy.shape[1]
	p_m = np.mean(p, axis=1)
	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a # now it should be correct

def uncertainty_ent_bays(probs, likelihoods): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = np.array(probs)
	# print("prob\n", probs)
	# print("likelihoods in bays", likelihoods)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	# print("entropy\n", entropy)

	a = np.sum(entropy, axis=2)
	al = a * likelihoods
	a = np.sum(al, axis=1)

	given_axis = 1
	dim_array = np.ones((1,probs.ndim),int).ravel()
	dim_array[given_axis] = -1
	b_reshaped = likelihoods.reshape(dim_array)
	mult_out = probs*b_reshaped
	p_m = np.sum(mult_out, axis=1)

	# p_m = np.mean(p, axis=1) #* likelihoods

	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a

def uncertainty_ent_bays2(probs, likelihoods): # idea from Willem
	p = np.array(probs)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)

	a = np.sum(entropy, axis=2)
	al = a * likelihoods
	a = np.sum(al, axis=1)

	e = (al.transpose() - a) ** 2
	e = e.transpose()
	e = np.mean(e, axis=1) / probs.shape[2]

	total = e + a
	return total, e, a


def uncertainty_ent_levi(probs, credal_size=30): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = [] #np.array(probs)
	for data_point in probs:
		d_p = []
		for sampling_seed in range(credal_size):
			d_p.append(resample(data_point, random_state=sampling_seed))
		p.append(np.array(d_p))
	p = np.array(p)
	p = np.mean(p, axis=2)
	entropy = -p*np.ma.log10(p)
	entropy = entropy.filled(0)
	a = np.sum(entropy, axis=1)
	a = np.sum(a, axis=1) / entropy.shape[1]
	p_m = np.mean(p, axis=1)
	total = -np.sum(p_m*np.ma.log10(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a # now it should be correct


def uncertainty_ent_standard(probs): # for tree
	p = np.array(probs)
	entropy = -p*np.ma.log10(p)
	entropy = entropy.filled(0)
	total = np.sum(entropy, axis=1)
	return total, total, total # now it should be correct


################################################################################################################################################# outcome uncertainty

def max_p_out_binary(s, p, l):
	s = np.reshape(s,(-1,1))
	l = np.reshape(l,(-1,1))
	s_l_p = s * l * p
	s_l = s * l
	s_l_p_sum = np.sum(s_l_p, axis=0)
	s_l_sum = np.sum(s_l, axis=0)
	z = s_l_p_sum / s_l_sum # to get the formual for S^* in paper
	z = z[0] # to only look at the first class of a binary problem
	return z * -1 # so that we maximize it

def min_p_out_binary(s, p, l):
	s = np.reshape(s,(-1,1))
	l = np.reshape(l,(-1,1))
	s_l_p = s * l * p
	s_l = s * l
	s_l_p_sum = np.sum(s_l_p, axis=0)
	s_l_sum = np.sum(s_l, axis=0)
	z = s_l_p_sum / s_l_sum # to get the formual for S^* in paper
	z = z[0] # to only look at the first class of a binary problem
	return z 

def uncertainty_outcome(probs, likelyhoods, epsilon=2, log=False):
	m  = len(likelyhoods)
	_m = 1/m

	cons = ({'type': 'eq', 'fun': constarint})
	b = (_m * (1 / epsilon), _m * epsilon) # (_m - epsilon, _m + epsilon) addetive constraint
	bnds = [ b for _ in range(m) ]
	x0 = get_random_with_constraint(probs.shape[1],bnds)

	s_max = []
	s_min = []
	for data_point_prob in probs:	
		sol_min = minimize(min_p_out_binary, x0, args=(data_point_prob,likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
		sol_max = minimize(max_p_out_binary, x0, args=(data_point_prob,likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
		s_min.append(sol_min.fun)
		s_max.append(-sol_max.fun)
	
	a = np.array(s_min)
	b = np.array(s_max)

	eu = b - a
	au = np.minimum(a, 1-b)
	tu = np.minimum(1-a, b)
	
	if log:
		print(probs)
		print("------------------------------------a")
		print(a)
		print("------------------------------------b")
		print(b)
		print("------------------------------------unc")
		print(eu)
		print(au)
		print(tu)
		exit()
	return tu, eu, au


def uncertainty_outcome_tree(probs, log=False):
	print("Yes we are in the tree!!!!")
	# what is a and b -> lower and upper bound of the credal set -> highest and lowest prob of a class
	a = probs[:,:,0].min(axis=1)
	b = probs[:,:,0].max(axis=1)

	eu = b - a
	au = np.minimum(a, 1-b)
	tu = np.minimum(1-a, b)
	
	if log:
		print(probs)
		print("------------------------------------a")
		print(a)
		print("------------------------------------b")
		print(b)
		print("------------------------------------unc")
		print(eu)
		print(au)
		print(tu)
	return tu, eu, au



################################################################################################################################################# set

def Interval_probability(total_number, positive_number):
    s = 1
    eps = .001
    valLow = (total_number - positive_number + s*eps*.5)/(positive_number+ s*(1-eps*.5))
    valUp = (total_number - positive_number + s*(1-eps*.5))/(positive_number+ s*.5)
    return [1/(1+valUp), 1/(1+valLow)]

def uncertainty_credal_point(total_number, positive_number):
    lower_probability, upper_probability = Interval_probability(total_number, positive_number)
    # due to the observation that in binary classification 
    # lower_probability_0 = 1-upper_probability_1 
    # upper_probability_0 = 1-lower_probability_1
    return -max(lower_probability/(1-lower_probability),(1-upper_probability)/upper_probability) 

def uncertainty_credal_tree(counts):
	credal_t = np.zeros(len(counts))
	for i, count in enumerate(counts):
		# print(f"index {i} count : {count[0][0]}     {count[0][1]}")
		credal_t[i] = uncertainty_credal_point(count[0][0] + count[0][1], count[0][1])
	return credal_t, credal_t, credal_t

def uncertainty_credal_tree_DF(counts):
	print(counts.shape)
	exit()

	credal_t = np.zeros(len(counts))
	for i, count in enumerate(counts):
		# print(f"index {i} count : {count[0][0]}     {count[0][1]}")
		credal_t[i] = uncertainty_credal_point(count[0][0] + count[0][1], count[0][1])
	return credal_t, credal_t, credal_t


def findallsubsets(s):
    res = np.array([])
    for n in range(1,len(s)+1):
        res = np.append(res, list(map(set, itertools.combinations(s, n))))
    return res

def v_q(set_slice):
	# print("------------------------------------v_q 14")
	# print(set_slice)
	# print(set_slice.shape)
	# sum
	sum_slice = np.sum(set_slice, axis=2)
	# min
	# print("------------------------------------ sum_slice")
	# print(sum_slice)
	# print(sum_slice.shape)

	min_slice = np.min(sum_slice, axis=1)
	# print("------------------------------------ min_slice")
	# print(min_slice)
	# print(min_slice.shape)
	return min_slice

def m_q(probs):
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # this is B in the paper
	set_A = subsets[-1]


	for set_B in subsets:
		set_slice = probs[:,:,list(set_B)]
		set_minus = set_A - set_B
		m_q_set = v_q(set_slice) * ((-1) ** len(set_minus))
		# print(f">>> {set_B}		 {m_q_set}")
		res += m_q_set
	return res

def set_gh(probs):
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # these subests are A in the paper
	# print("All subsets in GH ",subsets)

	for subset in subsets:
		set_slice = probs[:,:,list(subset)]
		m_q_slice = m_q(set_slice)
		res += m_q_slice * math.log2(len(subset))
	return res

def uncertainty_set14(probs, bootstrap_size=0, sampling_size=0, credal_size=0, log=False):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs
		
	if log:
		print("------------------------------------set14 prob after averaging each ensemble")
		print("Set14 p \n" , p)
		print(p.shape)
	# entropy = -p*np.log2(p)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)

	entropy_sum = np.sum(entropy, axis=2)
	s_max = np.max(entropy_sum, axis=1)
	s_min = np.min(entropy_sum, axis=1)
	gh    = set_gh(p)
	total = s_max
	e = gh
	a = total - e
	return total, e, a 


def uncertainty_set15(probs, bootstrap_size=0, sampling_size=0, credal_size=0):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			# print("number_of_samples ", number_of_samples)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs

	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	entropy_sum = np.sum(entropy, axis=2)
	s_min = np.min(entropy_sum, axis=1)
	s_max = np.max(entropy_sum, axis=1)
	total = s_max
	e = s_max - s_min
	a = total - e
	return total, e, a 

def uncertainty_set16(probs, bootstrap_size=0, sampling_size=0, credal_size=0, log=False):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs
		
	if log:
		print("------------------------------------set14 prob after averaging each ensemble")
		print("Set14 p \n" , p)
		print(p.shape)
	# entropy = -p*np.log2(p)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)

	entropy_sum = np.sum(entropy, axis=2)
	s_min = np.min(entropy_sum, axis=1)
	gh    = set_gh(p)
	e = gh
	a = s_min
	total = a + e

	return total, e, a 

def uncertainty_set17(probs, bootstrap_size=0, sampling_size=0, credal_size=0, log=False):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs
		
	if log:
		print("------------------------------------set14 prob after averaging each ensemble")
		print("Set14 p \n" , p)
		print(p.shape)
	# entropy = -p*np.log2(p)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	p_m = np.mean(p, axis=1)
	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	entropy_sum = np.sum(entropy, axis=2)
	s_max = np.max(entropy_sum, axis=1)

	gh    = set_gh(p)
	e = gh
	a = s_max
	total = a + e

	return total, e, a 


def v_q_a(s, l, p):
	# print(" the prob in the min opt ", p.shape)
	s = np.reshape(s,(-1,1))
	l = np.reshape(l,(-1,1))
	# print("------------------------------------ [start of V_Q_A]")

	# print("s\n",s)
	# print("l\n",l)
	# print("p\n",p)

	# print("------------------------------------")
	s_l_p = s * l * p
	s_l = s * l

	# print(s_l_p)
	# print(s_l)
	# print("------------------------------------first sum for m")
	s_l_p_sum  = np.sum(s_l_p, axis=0)
	# print("s_l_p_sum ", s_l_p_sum)
	# print("------------------------------------second sum for j")
	s_l_p_j_sum = np.sum(s_l_p_sum, axis=0)
	# print("s_l_p_j_sum ", s_l_p_j_sum)

	s_l_sum = np.sum(s_l)
	# print(s_l_sum)
	z = s_l_p_j_sum / s_l_sum 
	# print("------------------------------------final value")
	# print(z)
	# print("------------------------------------ [end of V_Q_A]")
	return z

def v_q18_2(set_slice, likelyhoods, epsilon):
	# print(" >>>>>>>>>>>>>>>>>>>>>>>> v_q ")
	m  = len(likelyhoods)
	_m = 1/m

	# print("set_slice\n", set_slice)
	# print("likelyhoods\n", likelyhoods)
	# print("------------------------------------")

	cons = ({'type': 'eq', 'fun': constarint})
	# b = (_m * (1 / epsilon), _m * epsilon) # (_m - epsilon, _m + epsilon) addetive constraint
	b = (1 / (m * epsilon), epsilon/m) # The bound as it is in the paper. The line above does not match with the bounds in the paper
	bnds = [ b for _ in range(m) ]
	# x0 = get_random_with_constraint(set_slice.shape[1],bnds)
	x0 = np.ones((set_slice.shape[1]))
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum
	# print(f"set_slice v_q {set_slice}")
	# print(f"x0 {x0}")


	s_min = []
	for data_point_prob in set_slice:	
		sol_min = minimize(v_q_a, x0, args=(likelyhoods,data_point_prob), method='SLSQP', bounds=bnds, constraints=cons)
		s_min.append(sol_min.fun)

		# print(f"opt min {sol_min.fun}")

		####### sanity test on S^* and S_*
		
		# for i in range(100):
		# 	# generate random s
		# 	rand_s = get_random_with_constraint(set_slice.shape[1],bnds)
		# 	# ent_rand = calculete convex_ent_max18 with random s
		# 	rand_v_q_a = v_q_a(rand_s, likelyhoods, data_point_prob)
		# 	# compare 
		# 	if rand_v_q_a < sol_min.fun:
		# 		print(f">>>>>>>>> [Failed] the test {i} rand_v_q_a {rand_v_q_a} min_v_q_a {sol_min.fun} ")
		# 	else:
		# 		print(f"pass rand_v_q_a {rand_v_q_a}  min_v_q_a {sol_min.fun}")

		####### end test (test passed)

	# print(" >>>>>>>>>>>>>>>>>>>>>>>> v_q end")

	res = np.array(s_min)
	return res

def v_q18(set_slice, likelyhoods, epsilon):
	print("------------------------------------start v_q18 set_slice.shape\n", set_slice)
	print(likelyhoods)
	m  = len(likelyhoods)
	_m = 1/m

	# sum_slice = np.sum(set_slice, axis=2) # to sum over all subsets for j in J in V_Q equation of the paper
	c_zeros = np.zeros((set_slice.shape[0],set_slice.shape[2]))


	given_axis = 1
	dim_array = np.ones((1,set_slice.ndim),int).ravel()
	dim_array[given_axis] = -1
	b_reshaped = likelyhoods.reshape(dim_array)
	lp = set_slice*b_reshaped
	# p_m = np.sum(mult_out, axis=1)
	print("------------------------------------")
	print(lp)
	c_alldata = np.concatenate((lp, c_zeros), axis=1)  # c is l*p sumed up for every j in J and then extented to c' by adding 0 for t (transformation of the LFP to LP)
	print(c_alldata)
	exit()
	d = np.concatenate((likelyhoods, [0]), axis=0)
	d = np.reshape(d, (1,-1))

	A = np.zeros((2*m+2, m+1)) # A' in wiki transform. it includes -b
	A[0,:] = 1
	A[0,-1] = -1
	A[1,:] = -1
	A[1,-1] = 1

	for i in range(2,2*m+2):
		for j in range(m+1):
			if (i-2)/2 == j:
				A[i,j]    = 1
				A[i,-1]   = -1 * _m * epsilon # this value is multiplied by -1 becase b is inside A as -b # -1*_m - epsilon (the addetive constraint)
				A[i+1,j]  = -1
				A[i+1,-1] = _m * (1/ epsilon) # the same is true for the lower bound # _m - epsilon (the addetive constraint)
		i = i+1

	b_ub = np.zeros(2*m+2)
	b_eq = np.ones((1))
	bounds = [ (None, None) for _ in range(m+1)]
	bounds[-1] = (0, None) # this is for t>=0 in LFP to LP transformation

	func_min = []
	for c in c_alldata: # c is a single datapoint c
		res = linprog(c, A_ub=A, b_ub=b_ub, A_eq=d, b_eq=b_eq, bounds=bounds, method='revised simplex')
		y = np.delete(res.x,-1)
		t = res.x[-1]
		x = y / t
		
		cc = np.delete(c,-1)
		dd = np.delete(d,-1)

		func_value = (cc*x) / (dd*x)
		func_value = func_value.sum() # this is to sum up all the hyposesies from m=1 to M

		########## Sanity test for V_Q(A)

		bnds = [ (_m * (1 / epsilon), _m * epsilon) for _ in range(m) ]
		for i in range(100):
			# generate random s
			rand_s = get_random_with_constraint(m,bnds)
			# ent_rand = calculete V_Q(A) with random s
			rand_vqa = (cc*rand_s) / (dd*rand_s)
			rand_vqa = rand_vqa.sum() # this is to sum up all the hyposesies from m=1 to M

			# compare
			if rand_vqa < func_value:
				func_value = rand_vqa
				# print(f">>>>>>>>> [Failed] the test {i} rand_vqa {rand_vqa} min_vqa {func_value}  diff {func_value - rand_vqa}")
			else:
				pass
				# print(f"pass rand_vqa {rand_vqa} min_vqa {func_value}")


		########## end test

		func_min.append(func_value)
	res = np.array(func_min)
	# print("------------------------------------end v_q18")

	return res

def m_q18(probs, likelyhoods, epsilon):
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # this is B in the paper
	set_A = subsets[-1]

	# print("######################################## m_q")
	# print(f"probs {probs.shape}")
	# print(f"subsets {subsets}")
	for set_B in subsets:
		set_slice = probs[:,:,list(set_B)]
		# print(f"set_slice m_q {set_slice.shape}")
		set_minus = set_A - set_B
		# m_q_set = v_q18(set_slice, likelyhoods, epsilon) * ((-1) ** len(set_minus))
		m_q_set = v_q18_2(set_slice, likelyhoods, epsilon) * ((-1) ** len(set_minus))
		# print(f">>> {set_B}		 {m_q_set}")
		res += m_q_set
	# print("######################################## m_q end")
	return res

def set_gh18(probs, likelyhoods, epsilon):
	# print("[debug] >>>> ", probs.shape)
	
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # these subests are A in the paper
	# print("[debug] >>>> subsets  ",subsets)
	for subset in subsets:
		set_slice = probs[:,:,list(subset)]
		# print("[debug] >>>> slice  ", set_slice.shape)
		# dkjhekjbd
		m_q_slice = m_q18(set_slice, likelyhoods, epsilon)
		res += m_q_slice * math.log2(len(subset))
	return res


def convex_ent_max18(s, p, l):
	s = np.reshape(s,(-1,1))
	l = np.reshape(l,(-1,1))
	s_l_p = s * l * p
	s_l = s * l
	s_l_p_sum = np.sum(s_l_p, axis=0)
	s_l_sum = np.sum(s_l, axis=0)
	z = s_l_p_sum / s_l_sum # to get the formual for S^* in paper
	entropy = -z*np.log2(z)
	entropy_sum = np.sum(entropy)
	return entropy_sum * -1 # so that we maximize it

def convex_ent_min18(s, p, l):
	s = np.reshape(s,(-1,1))
	l = np.reshape(l,(-1,1))
	s_l_p = s * l * p
	s_l = s * l
	s_l_p_sum = np.sum(s_l_p, axis=0)
	s_l_sum = np.sum(s_l, axis=0)
	z = s_l_p_sum / s_l_sum # to get the formual for S^* in paper
	entropy = -z*np.log2(z)
	entropy_sum = np.sum(entropy)
	return entropy_sum # so that we minimize it


def get_random_with_constraint(size, bound, tries=10000):
	x = []
	b_array = np.array(bound)
	for i in range(tries):
		x = np.random.dirichlet(np.ones(size),size=1)
		x = x[0]
		if np.less_equal(x, b_array[:,1]).all() and np.greater_equal(x, b_array[:,0]).all():
			return x
	print(f"[Warning] Did not find a random x within the bounds in {tries} tries")
	return x


def maxent18(probs, likelyhoods, epsilon):
	m  = len(likelyhoods)
	_m = 1/m

	cons = ({'type': 'eq', 'fun': constarint})
	# b = (_m * (1 / epsilon), _m * epsilon) # (_m - epsilon, _m + epsilon) addetive constraint
	b = (1 / (m * epsilon), epsilon/m) # The bound as it is in the paper. The line above does not match with the bounds in the paper
	# print(f" epsilon {epsilon} GH bounds ", b)

	bnds = [ b for _ in range(m) ]
	# x0 = get_random_with_constraint(probs.shape[1],bnds)
	x0 = np.ones((probs.shape[1]))
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum


	s_max = []
	for data_point_prob in probs:	
		sol_max = minimize(convex_ent_max18, x0, args=(data_point_prob,likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)

		####### sanity test on S^* and S_*
		
		# sol_min = minimize(convex_ent_min18, x0, args=(data_point_prob,likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
		# if test==True:
		# 	for i in range(100):
		# 		# generate random s
		# 		rand_s = get_random_with_constraint(probs.shape[1],bnds)
		# 		# ent_rand = calculete convex_ent_max18 with random s
		# 		rand_ent = convex_ent_max18(rand_s, data_point_prob, likelyhoods) * -1
		# 		# compare with sol_max
		# 		# if ent_rand > sol_max print test failed
		# 		if rand_ent > -sol_max.fun or rand_ent < sol_min.fun:
		# 			print(f">>>>>>>>> [Failed] the test {i} rand_ent {rand_ent} max_ent {-sol_max.fun} min_ent {sol_min.fun} ")
		# 		else:
		# 			print(f"pass rand_ent {rand_ent} max_ent {-sol_max.fun} min_ent {sol_min.fun}")

		####### end test (test passed)

		s_max.append(-sol_max.fun)

	return np.array(s_max)

def minent19(probs, likelyhoods, epsilon):
	m  = len(likelyhoods)
	_m = 1/m

	cons = ({'type': 'eq', 'fun': constarint})
	# b = (_m * (1 / epsilon), _m * epsilon) # (_m - epsilon, _m + epsilon) addetive constraint
	b = (1 / (m * epsilon), epsilon/m) # The bound as it is in the paper. The line above does not match with the bounds in the paper

	bnds = [ b for _ in range(m) ]
	# x0 = get_random_with_constraint(probs.shape[1],bnds)
	x0 = np.ones((probs.shape[1]))
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum

	s_min = []
	for data_point_prob in probs:	
		sol_min = minimize(convex_ent_min18, x0, args=(data_point_prob,likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
		s_min.append(sol_min.fun)

	return np.array(s_min)


def uncertainty_set18(probs, likelyhoods, epsilon=2, log=False): # credal sets based on the idea of chainging the prior from uniform to delda
	gh = set_gh18(probs, likelyhoods, epsilon)
	s_max = maxent18(probs, likelyhoods, epsilon)

	total = s_max
	e = gh
	a = total - e
	return total, e, a 

def uncertainty_set19(probs, likelyhoods, epsilon=2, log=False):
	s_max = maxent18(probs, likelyhoods, epsilon)
	s_min = minent19(probs, likelyhoods, epsilon)

	total = s_max
	e = s_max - s_min
	a = s_min
	return total, e, a 

################################################################################################################################################# set30 same as set18 but without epsilon


def v_q_a30(p, l, class_index_list):
	
	# l = np.reshape(l,(-1,1))
	# l_p = l * p
	# l_p_sum  = np.sum(l_p, axis=0)
	# l_p_j_sum = l_p_sum[class_index_list].sum(axis=0) # np.sum(l_p_sum, axis=0)

	# l_sum = np.sum(l)
	# z = l_p_j_sum / l_sum 
	p_sum = p[class_index_list].sum(axis=0)

	return p_sum

def v_q30(subset_subset, likelyhoods, probs):
	cons = ({'type': 'eq', 'fun': constarint})
	gh_min = []
	for data_index, data_point_prob in enumerate(probs):	
		x0_index = random.randint(0,len(likelyhoods)-1)
		x0 = data_point_prob[x0_index]
		bnds = []
		for class_index in range(probs.shape[2]):
			b_min = 0
			b_max = 1
			if class_index in subset_subset:
				b_min = data_point_prob[:,class_index].min()
				b_max = data_point_prob[:,class_index].max()
			bnds.append((b_min, b_max))
		sol_min = minimize(v_q_a30, x0, args=(likelyhoods,subset_subset), method='SLSQP', bounds=bnds, constraints=cons)
		gh_min.append(sol_min.fun)
	res = np.array(gh_min)
	return res

def m_q30(class_subset, likelyhoods, probs):
	res = np.zeros(probs.shape[0]) # FFFFFFFIIIIIXXXXXX
	subsets = findallsubsets(class_subset) # this is B in the paper
	set_A = subsets[-1]

	for set_B in subsets:
		set_minus = set_A - set_B
		m_q_set = v_q30(list(set_B), likelyhoods ,probs) * ((-1) ** len(set_minus))
		res += m_q_set
	return res

def set_gh30(probs, likelyhoods):
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # these subests are A in the paper
	for subset in subsets:
		m_q_slice = m_q30(subset, likelyhoods, probs) # probs is only passed in to claculate the bounds later
		res += m_q_slice * math.log2(len(subset))
	return res



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
	return entropy_sum

def maxent30(probs, likelyhoods):
	cons = ({'type': 'eq', 'fun': constarint})
	s_max = []
	for data_point_prob in probs:	
		x0_index = random.randint(0,len(likelyhoods)-1)
		x0 = data_point_prob[x0_index]
		bnds = []
		for class_index in range(probs.shape[2]):
			b_min = data_point_prob[:,class_index].min()
			b_max = data_point_prob[:,class_index].max()
			bnds.append((b_min, b_max))
		sol_max = minimize(convex_ent_max30, x0, args=(likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
		s_max.append(-sol_max.fun)
	return np.array(s_max)

def minent31(probs, likelyhoods):
	cons = ({'type': 'eq', 'fun': constarint})
	s_min = []
	for data_point_prob in probs:	
		x0_index = random.randint(0,len(likelyhoods)-1)
		x0 = data_point_prob[x0_index]
		bnds = []
		for class_index in range(probs.shape[2]):
			b_min = data_point_prob[:,class_index].min()
			b_max = data_point_prob[:,class_index].max()
			bnds.append((b_min, b_max))
		sol_max = minimize(convex_ent_min30, x0, args=(likelyhoods), method='SLSQP', bounds=bnds, constraints=cons)
		s_min.append(sol_max.fun)
	return np.array(s_min)

############################################################################

def uncertainty_set30(probs, likelyhoods, log=False): # credal set with different hyper prameters
	# print(probs)
	gh = set_gh30(probs, likelyhoods)
	# print(gh)
	# gh = set_gh(probs) # non convex
	s_max = maxent30(probs, likelyhoods)

	total = s_max
	e = gh
	a = total - e
	return total, e, a 

def uncertainty_set31(probs, likelyhoods, epsilon=2, log=False):
	s_max = maxent30(probs, likelyhoods)
	s_min = minent31(probs, likelyhoods)

	total = s_max
	e = s_max - s_min
	a = s_min
	return total, e, a 

################################################################################################################################################# set mix

def uncertainty_setmix(probs, credal_size=30):
	p = [] #np.array(probs)
	for data_point in probs:
		d_p = []
		for sampling_seed in range(credal_size):
			d_p.append(resample(data_point, random_state=sampling_seed))
		p.append(np.array(d_p))
	p = np.array(p)
	p = np.mean(p, axis=2)

	entropy = -p*np.log2(p)
	entropy_sum = np.sum(entropy, axis=2)
	s_min = np.min(entropy_sum, axis=1)
	s_max = np.max(entropy_sum, axis=1)
	e = set_gh(p)
	a = s_max - (s_max - s_min)
	total = e + a
	return total, e, a 


################################################################################################################################################# set convex

def convex_ent_max(alpha, p):
	alpha = np.reshape(alpha,(-1,1))
	p_alpha = alpha * p
	p_alpha_sum = np.sum(p_alpha, axis=0)
	entropy = -p_alpha_sum*np.log2(p_alpha_sum)
	entropy_sum = np.sum(entropy)
	return entropy_sum * -1 # so that we maximize it

def convex_ent_min(alpha, p):
	alpha = np.reshape(alpha,(-1,1))
	p_alpha = alpha * p
	p_alpha_sum = np.sum(p_alpha, axis=0)
	entropy = -p_alpha_sum*np.log2(p_alpha_sum)
	entropy_sum = np.sum(entropy)
	return entropy_sum # so that we maximize it

def constarint(x):
    sum_x = np.sum(x)
    return 1 - sum_x

def uncertainty_set14_convex(probs,bootstrap_size=0):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	else:
		p = probs
	cons = ({'type': 'eq', 'fun': constarint})

	# m  = probs.shape[1]
	# _m = 1/m
	# epsilon = 1
	# b = (_m * (1 / epsilon), _m * epsilon)
	b = (0.0, 1.0)
	bnds = [ b for _ in range(probs.shape[1]) ]
	x0 = np.ones((probs.shape[1]))
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum

	s_max = []
	for data_point_prob in probs:	
		sol_max = minimize(convex_ent_max, x0, args=data_point_prob, method='SLSQP', bounds=bnds, constraints=cons)
		s_max.append(-sol_max.fun)


	s_max = np.array(s_max)
	gh    = set_gh(p)
	# gh = set_gh30(p, x0, 1)
	total = s_max
	e = gh
	a = total - e
	return total, e, a 

def uncertainty_set15_convex(probs, bootstrap_size=0):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	else:
		p = probs
	cons = ({'type': 'eq', 'fun': constarint})
	b = (0.0, 1.0)
	bnds = [ b for _ in range(probs.shape[1]) ]
	x0 = np.random.rand(probs.shape[1])
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum

	s_max = []
	s_min = []
	for data_point_prob in probs:	
		sol_max = minimize(convex_ent_max, x0, args=data_point_prob, method='SLSQP', bounds=bnds, constraints=cons)
		s_max.append(-sol_max.fun)
		sol_min = minimize(convex_ent_min, x0, args=data_point_prob, method='SLSQP', bounds=bnds, constraints=cons)
		s_min.append(sol_min.fun)
	
	s_max = np.array(s_max)
	s_min = np.array(s_min)
	total = s_max
	e = s_max - s_min
	a = s_min #total - e
	return total, e, a

################################################################################################################################################# GS agent


def uncertainty_gs(probs, likelyhoods, credal_size):
	sorted_index = np.argsort(likelyhoods, kind='stable')
	l = likelyhoods[sorted_index]
	p = probs[:,sorted_index]

	gs_total = []
	gs_epist = []
	gs_ale   = []
	for level in range(credal_size-1):
		p_cut = p[:,0:level+2] # get the level cut probs based on sorted likelyhood
		# computing levi (set14) for level cut p_cut and appeinding to the unc array
		entropy = -p_cut*np.ma.log2(p_cut)
		entropy = entropy.filled(0)
		entropy_sum = np.sum(entropy, axis=2)
		s_max = np.max(entropy_sum, axis=1)
		s_min = np.min(entropy_sum, axis=1)
		gh    = set_gh(p_cut)
		total = s_max
		e = gh
		a = total - e
		gs_total.append(total)
		gs_epist.append(e)
		gs_ale.append(a)

	gs_total = np.mean(np.array(gs_total), axis=0)	
	gs_epist = np.mean(np.array(gs_epist), axis=0)	
	gs_ale   = np.mean(np.array(gs_ale), axis=0)	

	return gs_total, gs_epist, gs_ale



################################################################################################################################################# rl

def unc_rl_prob(train_probs, pool_probs, y_train, log=False):

	log_likelihoods = []
	for prob in train_probs:
		l = y_train * prob
		l = np.sum(l, axis=1)
		l = np.log(l)
		l = np.sum(l)
		log_likelihoods.append(l)
	log_likelihoods = np.array(log_likelihoods)
	
	max_l = np.amax(log_likelihoods)
	normalized_likelihoods = np.exp(log_likelihoods - max_l)
	if log:
		print(">>> debug log_likelihoods \n", log_likelihoods)
		print(">>> debug normalized_likelihoods \n", normalized_likelihoods)
		print(">>> debug train_probs \n", train_probs[:,0,:])
		print("------------------------------------")
	min_pos_nl_list = []
	min_neg_nl_list = []
	for prob, nl in zip(pool_probs, normalized_likelihoods):
		# print(prob)
		pos = prob[:,0] - prob[:,1] # diffefence betwean positive and negetive class
		neg = prob[:,1] - prob[:,0] #-1 * pos # diffefence betwean negetive and positive class
		pos = pos.clip(min=0)
		neg = neg.clip(min=0)
		
		nl_array = np.full(pos.shape, nl) # the constant likelihood vector

		min_pos_nl_list.append(np.minimum(nl_array,pos)) # min for pos support
		min_neg_nl_list.append(np.minimum(nl_array,neg)) # min for neg support
	min_pos_nl_list = np.array(min_pos_nl_list)
	min_neg_nl_list = np.array(min_neg_nl_list)
	pos_suppot = np.amax(min_pos_nl_list, axis=0) # sup for pos support
	neg_suppot = np.amax(min_neg_nl_list, axis=0) # sup for neg support
	epistemic = np.minimum(pos_suppot, neg_suppot)
	aleatoric = 1 - np.maximum(pos_suppot, neg_suppot)
	total = epistemic + aleatoric

	return total, epistemic, aleatoric


def uncertainty_rl_avg(counts):
	# unc = np.zeros((counts.shape[0],counts.shape[1],3))
	support = np.zeros((counts.shape[0],counts.shape[1],2))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			# res = relative_likelihood(y[0],y[1])
			# if y[0] >= 1000 or y[1] >= 1000:
			res = degrees_of_support_linh(y[1]+y[0],y[0])
			# else:
			# res = linh_fast(y[0],y[1])
			# res = rl_fast(y[0],y[1])
			support[i][j] = res
			# unc[i][j] = res
	support = np.mean(support, axis=1)
	t, e, a = rl_unc(support)
	# unc = unc / np.linalg.norm(unc, axis=0)		
	# unc = np.mean(unc, axis=1)
	# t = unc[:,0]
	# e = unc[:,1]
	# a = unc[:,2]
	return t,e,a

def unc_avg_sup_rl(counts):
	support = np.zeros((counts.shape[0],counts.shape[1],2))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			# res = degree_of_support(y[0],y[1])
			# res = degrees_of_support_linh(y[1]+y[0],y[0])
			res = linh_fast(y[0],y[1])
			# res = sup_fast(y[0],y[1])
			support[i][j] = res
	support = np.mean(support, axis=1)
	t, e, a = rl_unc(support)
	return t,e,a

def unc_rl_score(counts):
	support = np.zeros((counts.shape[0],counts.shape[1],2))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			res = linh_fast(y[0],y[1])
			support[i][j] = res
	
	# unc calculation
	epistemic = np.minimum(support[:,:,0], support[:,:,1])
	aleatoric = 1 - np.maximum(support[:,:,0], support[:,:,1])
	total = epistemic + aleatoric

	epistemic = np.mean(epistemic, axis=1)
	aleatoric = np.mean(aleatoric, axis=1)
	total     = np.mean(total,     axis=1)

	# score calculation
	s_score = np.reshape(support,(-1,2))
	i1 = np.arange(s_score.shape[0])
	i2 = s_score.argmin(axis=1)
	s_score[i1, i2] = 0
	s_score = np.reshape(s_score,(-1,counts.shape[1],2))
	s_score = np.mean(s_score, axis=1)
	s_score  = np.abs(s_score[:,0] - s_score[:,1])
	s_score = 1 - s_score

	# final unc * score
	epistemic = s_score * epistemic
	aleatoric = s_score * aleatoric
	total     = s_score * total

	return total,epistemic,aleatoric

def EpiAle_Averaged_Uncertainty_Preferences_Weight(clf, X_train, y_train, X_pool, uncertype, n_trees):
    n_samples = len(y_train)
    length_pool = len(X_pool)
    uncertainties = [0 for ind in range(length_pool)]
    pool_preferencePos = [0 for ind in range(length_pool)]
    pool_preferenceNeg = [0 for ind in range(length_pool)]
    for tree in clf:
        acc = tree.score(X_train, y_train)
        leaves_indices = tree.apply(X_pool, check_input=True).tolist()
        unique_leaves_indices = list(set(leaves_indices))
        leaves_indices_train = tree.apply(X_train, check_input=True).tolist()
        leaves_inices_train_pos = [leaves_indices_train[i] for i in range(n_samples) if y_train[i] == 1]
        leaves_size_neighbours = []
        leaves_uncertainties = []
        leaves_preferencePos = []
        leaves_preferenceNeg = []
        for leaf_index in range(len(unique_leaves_indices)):
            n_total_instance = leaves_indices_train.count(unique_leaves_indices[leaf_index])
            n_positive_instance = leaves_inices_train_pos.count(unique_leaves_indices[leaf_index])
            leaves_size_neighbours.append(n_total_instance)
            posSupPa, negSupPa =  degrees_of_support_linh(n_total_instance, n_positive_instance)
            epistemic = min(posSupPa, negSupPa)
            aleatoric = 1 -max(posSupPa, negSupPa)
            if posSupPa > negSupPa:
                preferencePos = 1- (epistemic+aleatoric)
            elif posSupPa == negSupPa:
                preferencePos = 1- (epistemic+aleatoric)/2
            else:
                preferencePos = 0
            preferenceNeg = 1 - (epistemic+aleatoric+preferencePos)
            leaves_preferencePos.append(preferencePos*acc)
            leaves_preferenceNeg.append(preferenceNeg*acc)
            if uncertype == "e": # Epistemic uncertainty
                leaves_uncertainties.append(epistemic)
            if uncertype == "a": # Aleatoric uncertainty
                leaves_uncertainties.append(aleatoric)
            if uncertype == "t": # Epistemic + Aleatoric uncertainty
               leaves_uncertainties.append(epistemic + aleatoric)       
        for instance_index in range(length_pool):
            uncertainties[instance_index] += leaves_uncertainties[unique_leaves_indices.index(leaves_indices[instance_index])]
#            neighbour_sizes[instance_index] += [leaves_size_neighbours[unique_leaves_indices.index(leaves_indices[instance_index])]]
            pool_preferencePos[instance_index] +=leaves_preferencePos[unique_leaves_indices.index(leaves_indices[instance_index])]
            pool_preferenceNeg[instance_index] +=leaves_preferenceNeg[unique_leaves_indices.index(leaves_indices[instance_index])]
    uncertaintiesEA = []
    for instance_index in range(length_pool):
        AveragePreferencePos = pool_preferencePos[instance_index]/n_trees
        AveragePreferenceNeg = pool_preferenceNeg[instance_index]/n_trees
        score = 1 - abs(AveragePreferencePos-AveragePreferenceNeg)
        uncertaintiesEA.append(score*uncertainties[instance_index]/n_trees)
    return uncertaintiesEA

def EpiAle_Averaged_Uncertainty_Preferences(clf, X_train, y_train, X_pool, uncertype, n_trees):
    n_samples = len(y_train)
    length_pool = len(X_pool)
    positive_preferences = [0 for ind in range(length_pool)]
    negative_preferences = [0 for ind in range(length_pool)]
    uncertainty = [0 for ind in range(length_pool)]
    for tree in clf:
        acc = tree.score(X_train, y_train)
        leaves_indices = tree.apply(X_pool, check_input=True).tolist()
        unique_leaves_indices = list(set(leaves_indices))
        leaves_indices_train = tree.apply(X_train, check_input=True).tolist()
        leaves_inices_train_pos = [leaves_indices_train[i] for i in range(n_samples) if y_train[i] == 1]
        leaves_preferencePos = []
        leaves_preferenceNeg = []
        leaves_uncertainties = []
        for leaf_index in range(len(unique_leaves_indices)):
            n_total_instance = leaves_indices_train.count(unique_leaves_indices[leaf_index])
            n_positive_instance = leaves_inices_train_pos.count(unique_leaves_indices[leaf_index])
            posSupPa, negSupPa =  degrees_of_support_linh(n_total_instance, n_positive_instance)
            epistemic = min(posSupPa, negSupPa)
            aleatoric = 1 -max(posSupPa, negSupPa)
            if posSupPa > negSupPa:
                preferencePos = 1- (epistemic+aleatoric)
            elif posSupPa == negSupPa:
                preferencePos = 1- (epistemic+aleatoric)/2
            else:
                preferencePos = 0
            preferenceNeg = 1 - (epistemic+aleatoric + preferencePos)
            leaves_preferencePos.append(preferencePos*acc)
            leaves_preferenceNeg.append(preferenceNeg*acc)

            if uncertype == "e": # Epistemic uncertainty
                leaves_uncertainties.append(epistemic*acc)
            if uncertype == "a": # Aleatoric uncertainty
                leaves_uncertainties.append(aleatoric*acc) 
            if uncertype == "t": # Epistemic + Aleatoric uncertainty
               leaves_uncertainties.append((epistemic + aleatoric) *acc)
        for instance_index in range(length_pool):
            positive_preferences[instance_index] += leaves_preferencePos[unique_leaves_indices.index(leaves_indices[instance_index])]
            negative_preferences[instance_index] += leaves_preferenceNeg[unique_leaves_indices.index(leaves_indices[instance_index])]
            uncertainty[instance_index] += leaves_uncertainties[unique_leaves_indices.index(leaves_indices[instance_index])]
    uncertaintiesEA = []
    for instance_index in range(length_pool):
        preferencePos = positive_preferences [instance_index]/n_trees
        preferenceNeg = negative_preferences [instance_index]/n_trees
        score = 1 - abs(preferencePos-preferenceNeg)
        uncertaintiesEA.append((uncertainty[instance_index]/n_trees)*score)
    return uncertaintiesEA


def EpiAle_Averaged_Support_Uncertainty(clf, X_train, y_train, X_pool, uncertype, n_trees):
    n_samples = len(y_train)
    length_pool = len(X_pool)
    positive_support = [0 for ind in range(length_pool)]
    negative_support = [0 for ind in range(length_pool)]
    uncertainty = [0 for ind in range(length_pool)]
    for tree in clf:
        leaves_indices = tree.apply(X_pool, check_input=True).tolist()
        unique_leaves_indices = list(set(leaves_indices))
        leaves_indices_train = tree.apply(X_train, check_input=True).tolist()
        leaves_inices_train_pos = [leaves_indices_train[i] for i in range(n_samples) if y_train[i] == 1] 
        leaves_posSupPa = []
        leaves_negSupPa = []
        leaves_uncertainties = []
        for leaf_index in range(len(unique_leaves_indices)): 
            n_total_instance = leaves_indices_train.count(unique_leaves_indices[leaf_index])
            n_positive_instance = leaves_inices_train_pos.count(unique_leaves_indices[leaf_index])
            posSupPa, negSupPa =  degrees_of_support_linh(n_total_instance, n_positive_instance)
            if posSupPa >= negSupPa:
                leaves_posSupPa.append(posSupPa)
                leaves_negSupPa.append(0)
            else:
                leaves_posSupPa.append(0)
                leaves_negSupPa.append(negSupPa)
            if uncertype == "e": # Epistemic uncertainty
                leaves_uncertainties.append(min(posSupPa, negSupPa))
            if uncertype == "a": # Aleatoric uncertainty
                leaves_uncertainties.append(1 -max(posSupPa, negSupPa)) 
            if uncertype == "t": # Epistemic + Aleatoric uncertainty
               leaves_uncertainties.append(min(posSupPa, negSupPa) +1 -max(posSupPa, negSupPa))
        for instance_index in range(length_pool): 
            positive_support[instance_index] += leaves_posSupPa[unique_leaves_indices.index(leaves_indices[instance_index])]
            negative_support[instance_index] += leaves_negSupPa[unique_leaves_indices.index(leaves_indices[instance_index])]
            uncertainty[instance_index] += leaves_uncertainties[unique_leaves_indices.index(leaves_indices[instance_index])]
    uncertaintiesEA = []
    for instance_index in range(length_pool):
        posSupPa = positive_support[instance_index]/n_trees
        negSupPa = negative_support[instance_index]/n_trees
        score = 1 - abs(posSupPa-negSupPa)
        uncertaintiesEA.append((uncertainty[instance_index]/n_trees)*score) 
    return uncertaintiesEA


def uncertainty_rl_ALB(counts):
	unc = np.zeros((counts.shape[0],counts.shape[1],3))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			res = relative_likelihood(y[0],y[1])
			unc[i][j] = res
	e_unc = np.mean(unc, axis=1)
	a_unc = np.max(unc, axis=1)
	e = e_unc[:,1]
	a = a_unc[:,2]
	t = a + e

	return t,e,a



def uncertainty_rl_one(counts):
	unc = []
	for class_counts in counts:
		unc.append(relative_likelihood(class_counts[0],class_counts[1]))
	unc = np.array(unc)
	t = unc[:,0]
	e = unc[:,1]
	a = unc[:,2]
	return t,e,a
	

def likelyhood(p,n,teta):
	
	# old
	# a = teta**p
	# b = (1-teta)**n
	# c = (p/(n+p))**p
	# d = (n/(n+p))**n
	# return (a * b) / (c * d)

	if   p == 0:
		return ( ( (1-teta) * (n + p) ) / n ) ** n
	elif n == 0:
		return ( ( teta * (n + p) ) / p ) ** p
	else:
		return ( ( ( teta * (n + p) ) / p ) ** p ) * ( ( ( (1-teta) * (n + p) ) / n ) ** n )



def prob_pos(teta):
	return (2 * teta) - 1

def prob_neg(teta):
	return 1 - (2*teta)

def degree_of_support(pos,neg):
	sup_pos = 0
	sup_neg = 0
	for x in range(1,100):
		x /= 100

		l = likelyhood(pos,neg,x)
		p_pos = prob_pos(x)
		min_pos = min(l,p_pos)

		if min_pos > sup_pos:
			sup_pos = min_pos

		p_neg = prob_neg(x)

		min_neg = min(l,p_neg)
		if min_neg > sup_neg:
			sup_neg = min_neg
	return np.array([sup_pos, sup_neg])


############################################################ Linh
# def targetFunction(alpha, sizeins, posins, classId):
#     if classId == 1:
#        highFunc = max(2*alpha -1,0)
#     else:
#        highFunc = max(1 -2*alpha,0) 
#     necins = sizeins - posins
#     proportion = posins*(1/float(sizeins))
#     numerator = (alpha**posins)*((1-alpha)**necins)
#     denominator = (proportion**posins)*((1-proportion)**necins)
#     supportFunc = numerator*(1/float(denominator))
#     TargetFunc = - min(supportFunc, highFunc)
#     return TargetFunc
def targetFunction(alpha, sizeins, p, classId):
    if classId == 1:
        highFunc = max(2*alpha -1,0)
    else:
        highFunc = max(1 -2*alpha,0)
    n = sizeins - p
    if p == 0:
        supportFunc = (((1-alpha)*(n + p))/n)**n
    elif n == 0:
        supportFunc = ((alpha*(n+p))/p)**p
    else:
        supportFunc = (((alpha*(n+p))/p)**p)*((((1-alpha)*(n+p))/n)**n)
    TargetFunc = - min(supportFunc, highFunc)
    return TargetFunc
 
dictionary_DoS ={}    
def degrees_of_support_linh(sizeins, posins):
    global dictionary_DoS    
    key = "%i_%i"%(sizeins, posins) 
    if (key in dictionary_DoS):
        return dictionary_DoS.get(key)        
    if sizeins == 0:
        return [1,1]
    def Optp(alpha): return targetFunction(alpha, sizeins, posins, 1)
    posSupPa =  minimize_scalar(Optp, bounds=(0, 1), method='bounded')
    def Optn(alpha): return targetFunction(alpha, sizeins, posins, -1)
    negSupPa =  minimize_scalar(Optn, bounds=(0, 1), method='bounded')  
    dictionary_DoS[key] = [-posSupPa.fun, -negSupPa.fun] 
    return [-posSupPa.fun, -negSupPa.fun]
############################################################ Linh End


def rl_unc(support): # rl unc with the degrees of support
	epistemic = np.minimum(support[:,0], support[:,1])
	aleatoric = 1 - np.maximum(support[:,0], support[:,1])
	total = epistemic + aleatoric
	# unc = np.stack((total, epistemic, aleatoric), axis=1)
	return total, epistemic, aleatoric
	
def relative_likelihood(pos,neg):
	sup_pos = 0
	sup_neg = 0
	for x in range(1,1000):
		x /= 1000

		l = likelyhood(pos,neg,x)
		p_pos = prob_pos(x)
		min_pos = min(l,p_pos)

		if min_pos > sup_pos:
			sup_pos = min_pos

		p_neg = prob_neg(x)

		min_neg = min(l,p_neg)
		if min_neg > sup_neg:
			sup_neg = min_neg
	epistemic = min(sup_pos, sup_neg)
	aleatoric = 1 - max(sup_pos, sup_neg)
	total = epistemic + aleatoric

	# if pos ==0 and neg == 84:
	# 	print(f" pos {pos} neg {neg} unc {np.array([total, epistemic, aleatoric])}")


	return np.array([total, epistemic, aleatoric])

# n = 500
# rl_unc_array = np.zeros((n*2,n*2,3))
# rl_sup_array = np.zeros((n*2,n*2,2))
# rl_linh_array = np.zeros((n*2,n*2,2))

# for i in range(n):
# 	for j in range(i,n):
# 		if(i==0 and j==0):
# 			continue
# 		rl = relative_likelihood(i,j)
# 		rl_unc_array[i][j] = rl
# 		rl_unc_array[j][i] = rl

# for i in range(n):
# 	for j in range(n):
# 		if(i==0 and j==0):
# 			continue
# 		spd = degree_of_support(i,j)
# 		rl_sup_array[i][j] = spd

# for i in range(n):
# 	for j in range(n):
# 		if(i==0 and j==0):
# 			continue
# 		spd = degrees_of_support_linh(i + j,i)
# 		rl_linh_array[i][j] = spd

# with open('Data/pr_rl/rl_unc_array.npy', 'rb') as f:
# 	rl_unc_array = np.load(f)
# 	# print("rl_unc_array shape", rl_unc_array.shape)
# with open('Data/pr_rl/rl_sup_array.npy', 'rb') as f:
# 	rl_sup_array = np.load(f)
# 	# print("rl_sup_array shape", rl_sup_array.shape)
# with open('Data/pr_rl/rl_linh_array.npy', 'rb') as f:
# 	rl_linh_array = np.load(f)
# 	# print("rl_linh_array shape", rl_linh_array.shape)
rl_unc_array = 0
rl_sup_array = 0
rl_linh_array = 0
def rl_fast(pos,neg):	
	return rl_unc_array[int(pos)][int(neg)]

def sup_fast(pos,neg):	
	return rl_sup_array[int(pos)][int(neg)]

def linh_fast(pos,neg):	
	return rl_linh_array[int(pos)][int(neg)]




#################################################################################################################################################

def accuracy_rejection2(predictions_list, labels_list, uncertainty_list, log=False): # more accurate for calculating area under the curve
	accuracy_list = [] # list containing all the acc lists of all runs
	reject_list = []

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(-uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]

		accuracy = [] # list of all acc for a single run
		rejection = []
		for i in range(len(uncertainty)):
			t_correctness = correctness_map[i:]
			rej = (len(uncertainty) - len(t_correctness)) / len(uncertainty)
			acc = t_correctness.sum() / len(t_correctness)

			accuracy.append(acc)
			rejection.append(rej)
		accuracy_list.append(np.array(accuracy))
		reject_list.append(rejection)

	# print(">>>>>> ", reject_list)

	min_rejection_len = 999999999
	for rejection in reject_list:
		if len(rejection) < min_rejection_len:
			min_rejection_len = len(rejection)

	for i, (rejection, accuracy) in enumerate(zip(reject_list,accuracy_list)):
		reject_list[i] = rejection[:min_rejection_len]
		accuracy_list[i] = accuracy[:min_rejection_len]

	accuracy_list = np.array(accuracy_list)
	reject_list = np.array(reject_list, dtype=float)
		
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)

		avg_accuracy = np.nanmean(accuracy_list, axis=0)
		steps = np.nanmean(reject_list, axis=0)
		std_error = np.std(accuracy_list, axis=0) / math.sqrt(len(uncertainty_list))

	return avg_accuracy, avg_accuracy - std_error, avg_accuracy + std_error, 9999 , steps*100

def unc_heat_map(predictions_list, labels_list, epist_list, ale_list, log=False):
	unc = np.array(epist_list)
	heat_all = np.zeros((unc.shape[0], unc.shape[1], unc.shape[1]))
	rej = np.zeros((2, unc.shape[1]))

	run_index = 0
	for predictions, epist, ale, labels in zip(predictions_list, epist_list, ale_list, labels_list):
		
		predictions = np.array(predictions)
		epist = np.array(epist)
		ale = np.array(ale)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		correctness_map    = np.array(correctness_map)
		sorted_index_epist = np.argsort(-epist, kind='stable')
		sorted_index_ale   = np.argsort(-ale, kind='stable')

		for i in range(len(epist)):

			sorted_index_epist_t = sorted_index_epist[i:]
			
			for j in range(len(ale)):
				sorted_index_ale_t = sorted_index_ale[j:] # filter based on aleatoric uncertainty
				intersection_index = np.intersect1d(sorted_index_ale_t, sorted_index_epist_t) # intersection of ale and epist
				t_correctness = correctness_map[intersection_index]
				acc = t_correctness.sum() / len(t_correctness)
				heat_all[run_index][len(ale)-1-j][i] = acc
				# rej[0][j] =  j / len(ale)  # rejection percentage
				# rej[1][i] = (len(ale) - i) / len(ale)
				rej[0][i] =  epist[sorted_index_epist[i]] # uncertainty value
				rej[1][j] = ale[sorted_index_ale[len(ale) - j - 1]]
		run_index += 1
	heat = np.mean(heat_all, axis=0)
	# rej = rej * 100
	rej = np.round(rej, 5)
	return heat, rej

def order_comparison(uncertainty_list1, uncertainty_list2, log=False):
	tau_list = []
	pvalue_list = []
	for unc1, unc2 in zip(uncertainty_list1, uncertainty_list2):
		unc1 = np.array(unc1)
		unc2 = np.array(unc2)
		sorted_index1 = np.argsort(-unc1, kind='stable')
		sorted_index2 = np.argsort(-unc2, kind='stable')
		# unc1 = unc1[sorted_index1]
		# unc2 = unc2[sorted_index2]
		# tau, p_value = stats.kendalltau(sorted_index1, sorted_index2)
		tau, p_value = stats.kendalltau(unc1, unc2)
		# tau, p_value = stats.spearmanr(sorted_index1, sorted_index2)

		if log:
			print(sorted_index1)
			print(sorted_index2)
			print(unc1)
			print(unc2)
			print(f"{tau} pvalue {p_value}")
			print("------------------------------------")
			exit()
		tau_list.append(tau)
		pvalue_list.append(p_value)
	comp = mean(tau_list)
	comp_p = mean(pvalue_list)
	if log:
		print(f">>>>>>>>>>>>>>>>>>>>>  {comp} pvalue {comp_p}")
	return comp, comp_p

def accuracy_rejection(predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	accuracy_list = []
	r_accuracy_list = []
	
	steps = np.array(list(range(90)))
	if unc_value:
		steps = uncertainty_list


	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		# uncertainty, correctness_map = zip(*sorted(zip(uncertainty,correctness_map),reverse=False))

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]

		correctness_map = list(correctness_map)
		uncertainty = list(uncertainty)
		data_len = len(correctness_map)
		accuracy = []

		for step_index, x in enumerate(steps):
			if unc_value:
				rejection_index = step_index
			else:
				rejection_index = int(data_len *(len(steps) - x) / len(steps))
			x_correct = correctness_map[:rejection_index].copy()
			x_unc = uncertainty[:rejection_index].copy()
			if log:
				print(f"----------------------------------------------- rejection_index {rejection_index}")
				for c,u in zip(x_correct, x_unc):
					print(f"correctness_map {c} uncertainty {u}")
				# print(f"rejection_index = {rejection_index}\nx_correct {x_correct} \nunc {x_unc}")
			if rejection_index == 0:
				accuracy.append(np.nan) # random.random()
			else:
				accuracy.append(np.sum(x_correct) / rejection_index)
		accuracy_list.append(accuracy)

		# random test plot
		r_accuracy = []
		
		for step_index, x in enumerate(steps):
			random.shuffle(correctness_map)
			if unc_value:
				r_rejection_index = step_index
			else:
				r_rejection_index = int(data_len *(len(steps) - x) / len(steps))

			r_x_correct = correctness_map[:r_rejection_index].copy()
			if r_rejection_index == 0:
				r_accuracy.append(np.nan)
			else:
				r_accuracy.append(np.sum(r_x_correct) / r_rejection_index)

		r_accuracy_list.append(r_accuracy)

	accuracy_list = np.array(accuracy_list)
	r_accuracy_list = np.array(r_accuracy_list)
		
	# print(accuracy_list)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)

		avg_accuracy = np.nanmean(accuracy_list, axis=0)
		avg_r_accuracy = np.nanmean(r_accuracy_list, axis=0)
		std_error = np.std(accuracy_list, axis=0) / math.sqrt(len(uncertainty_list))


	return avg_accuracy, avg_accuracy - std_error, avg_accuracy + std_error, avg_r_accuracy , steps

def roc(probs_list, predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	area_list = []

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(0) 
			else:
				correctness_map.append(1)
		correctness_map = np.array(correctness_map)

		# probs = np.array(probs)
		# predictions = np.array(predictions)
		# uncertainty = np.array(uncertainty)
		# labels = np.array(labels)

		# fpr, tpr, thresholds = metrics.roc_curve(correctness_map, uncertainty)
		# area = metrics.auc(tpr, fpr)
		if len(np.unique(correctness_map)) == 1:
			# print(correctness_map)
			# print("Skipping")
			continue
		area = metrics.roc_auc_score(correctness_map, uncertainty)
		area_list.append(area)

	area_list = np.array(area_list)
	AUROC_mean = area_list.mean()
	AUROC_std  = area_list.std()

	return AUROC_mean, AUROC_std * 2

def roc_epist(probs_list, predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	area_list = []

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		# correctness_map = []
		# for x, y in zip(predictions, labels):
		# 	if x == y:
		# 		correctness_map.append(0) 
		# 	else:
		# 		correctness_map.append(1)
		correctness_map = np.array(labels)

		if len(np.unique(correctness_map)) == 1:
			continue
		# print("------------------------------------ unc and correctness_map shape")
		# print(correctness_map)
		area = metrics.roc_auc_score(correctness_map, uncertainty)
		area_list.append(area)

	area_list = np.array(area_list)
	AUROC_mean = area_list.mean()
	AUROC_std  = area_list.std()

	return AUROC_mean, AUROC_std * 2

def uncertainty_correlation(predictions_list, labels_list, uncertainty_list, log=False): # more accurate for calculating area under the curve
	corr_list = [] # list containing all the acc lists of all runs

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(0) # switching the correctness labels just to get positive corr values
			else:
				correctness_map.append(1)

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(-uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]
		count = np.unique(correctness_map)
		if len(count) == 1:
			continue
		corr = stats.pearsonr(uncertainty, correctness_map)
		if log:
			print(f"correctness_map \n{correctness_map.shape} uncertainty \n{uncertainty.shape} \ncorr {corr}")
		corr_list.append(corr)

	corr_list = np.array(corr_list)
	avg_corr = np.nanmean(corr_list, axis=0)
	return avg_corr

def uncertainty_distribution(predictions_list, labels_list, uncertainty_list, log=False): # more accurate for calculating area under the curve
	corr_list = [] # list containing all the acc lists of all runs
	unc_correct_all = np.array([])
	unc_incorrect_all = np.array([])

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1) 
			else:
				correctness_map.append(0)
		
		# sort based on correctness_map

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(-correctness_map, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]
		count = np.unique(correctness_map)

		# split correct from incorrect
		split_index = 0 # code to find where to split the correctness array
		for i, v in enumerate(correctness_map):
			if v != 1:
				split_index = i
				break
		corrects      = correctness_map[:split_index]
		unc_correct   = uncertainty[:split_index]
		incorrects    = correctness_map[split_index:]
		unc_incorrect = uncertainty[split_index:]
		unc_correct_all = np.concatenate((unc_correct_all, unc_correct))
		unc_incorrect_all = np.concatenate((unc_incorrect_all, unc_incorrect))
	
	return unc_correct_all, unc_incorrect_all
