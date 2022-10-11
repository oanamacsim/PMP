import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

import random

#calculez timpul total necesar pt un request x 10000(nr val random)luand in calcul timpul de raspuns al serverului
#la care adaugam latenta care e distributie exponentiala (asa am inteles la seminar)
server_1 = stats.gamma.rvs(4, scale=1 / 3, size=10000) + stats.expon.rvs(0, 1 / 4, size=10000)
server_2 = stats.gamma.rvs(4, scale=1 / 2, size=10000) + stats.expon.rvs(0, 1 / 4, size=10000)
server_3 = stats.gamma.rvs(5, scale=1 / 2, size=10000) + stats.expon.rvs(0, 1 / 4, size=10000)
server_4 = stats.gamma.rvs(5, scale=1 / 3, size=10000) + stats.expon.rvs(0, 1 / 4, size=10000)

# aleg 10000 valori random pe care i le atribui fiecarui server (1-4) conform probabilitatilor
random_value = random.choices((1, 2, 3, 4), weights=(0.25, 0.25, 0.30, 0.20), k=10000)

time_necessary = []
over_3_millisec = 0

#creez un array cu timpurile de procesare a requestului pt fiecare din cele 10000 de variante
for i in range(10000):
    if random_value[i] == 1:
        time_necessary.append(server_1[i])
    elif random_value[i] == 2:
        time_necessary.append(server_2[i])
    elif random_value[i] == 3:
        time_necessary.append(server_3[i])
    elif random_value[i] == 4:
        time_necessary.append(server_4[i])

#numar cate dureaza mai mult de 3 milisecunde pt calculul probabilitatii
for i in range(10000):
    if time_necessary[i] > 3:
        over_3_millisec += 1

print("Probability to have a request processing time over 3 milliseconds is:", (over_3_millisec/10000))

az.plot_posterior({"Mean:": time_necessary})
plt.show()