import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()
client = pm.Model()

with client:
    trafic = poisson.pmf('T', mu = 20)
    timp_plasare = []
    timp_comanda = []

    for i in range (trafic):
        timp_plasare += np.random.normal('TP', 1, 0.5)
        alfa = random() < .3
        timp_gatire += pm.Exponential('TG', alfa)
    trace = pm.sample(20000, chains = 1)

dictionary = {
              'trafic': trace['T'].tolist(),
              'timp_plasare': trace['TP'].tolist(),
              'timp_gatire': trace['TG'].tolist()
              }
df = pd.DataFrame(dictionary)
