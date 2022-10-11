import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

import random

ss = sb = bb = bs = 0

for i in range(100):
    fair_coin = random.choices((1, 2), weights = (0.5, 0.5), k = 10) 
    loaded_coin = random.choices((1, 2), weights = (0.3, 0.7), k = 10) 
    for j in range(10):
        if(fair_coin[j] == 1 and loaded_coin[j] == 1):
            ss += 1
        elif(fair_coin[j] == 1 and loaded_coin[j] == 2):
            sb += 1
        elif(fair_coin[j] == 2 and loaded_coin[j] == 2):
            bb += 1
        elif(fair_coin[j] == 2 and loaded_coin[j] == 1):
            bs += 1

az.plot_posterior({'ss': ss, 'sb': sb, 'bb': bb, 'bs': bs })
plt.show()