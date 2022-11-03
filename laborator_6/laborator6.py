import csv
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#1.
x = []
y = []
z = []
with open("data.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
     x.append(row[3])
     y.append(row[1])
     z.append(row[2])

_, ax = plt.subplots(1, 2, figsize=(12, 10))
ax[0].plot(x, y, 'C0.')
ax[0].set_xlabel('Mom age')
ax[0].set_ylabel('PPVT', rotation=0)
plt.tight_layout()
plt.show()

#2.
#standard dev
standard_dev = stats.stdev(y)
standard_dev2 = stats.stdev(x)
with pm.Model() as model_g:
    alfa = pm.Normal('alfa', mu=0, sd=10 * standard_dev)
    beta = pm.Normal('beta', mu=0, sd=1*standard_dev/standard_dev2)
    epsilon = pm.HalfCauchy('epsilon', 5)
    miu = pm.Deterministic('miu', alfa + beta * x)
    y_pred = pm.Normal('y_pred', mu=miu, sd=epsilon, observed=y)

idata_g = pm.sample(400, tune=400, return_inferencedata=True)
az.plot_trace(idata_g, var_names=["α", "β", "ε"])

#3.
alpha_m = alfa.mean().item()
beta_m = beta.mean().item()
ppc = pm.sample_posterior_predictive(idata_g, samples=400, model=model_g)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k',
label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hdi(x, ppc['y_pred'], hdi_prob=0.5, color='gray', smooth=False)
az.plot_hdi(x, ppc['y_pred'], color='gray', smooth=False)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

#4.
#st_Dev(de educatie) in model in loc de st_dev

standard_dev3 = stats.stdev(z)
with pm.Model() as model_g:
    alfa = pm.Normal('alfa', mu=0, sd=10 * standard_dev3)
    beta = pm.Normal('beta', mu=0, sd=1*standard_dev3/standard_dev2)
    epsilon = pm.HalfCauchy('epsilon', 5)
    miu = pm.Deterministic('miu', alfa + beta * x)
    y_pred = pm.Normal('y_pred', mu=miu, sd=epsilon, observed=y)

idata_g = pm.sample(400, tune=400, return_inferencedata=True)
az.plot_trace(idata_g, var_names=["α", "β", "ε"])
#