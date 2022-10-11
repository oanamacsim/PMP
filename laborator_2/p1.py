import numpy as np
from scipy import stats
from random import *

import matplotlib.pyplot as plt
import arviz as az

z = []

for i in range(10000):
    y = randint(1, 100);
    if y <= 40:
        m = stats.expon.rvs(0, 0.25, 1)
    else:
        m = stats.expon.rvs(0, 0.16, 1)
    z.append(m)


az.plot_posterior({'z':z}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 