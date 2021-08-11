import numpy as np
import random

pram_grid = {
    "max_depth" : np.arange(1,20), # [2,3,5,7,10,12,15,18,20]
    "criterion" : ["gini", "entropy"]
}

# keys = random.sample(list(pram_grid), 2)
# values = [pram_grid[k] for k in keys]
# print(values)

def sample_pram(pram_dict):
    sample = {}
    for pram in pram_dict:
        value = random.choice(pram_dict[pram])
        sample[pram] = value
    return sample
s = sample_pram(pram_grid)
