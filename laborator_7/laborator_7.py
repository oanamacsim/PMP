import csv
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

y = []
x1 = []
x2 = []

with open("Prices.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
     y.append(row[1])
     x1.append(row[2])
     x2.append(row[3])

standard_dev_y = stats.stdev(y)
standard_dev_x1 = stats.stdev(x1)
standard_dev_x2 = stats.stdev(x2)

with pm.Model() as model_g:
    alfa = pm.Normal('alfa', mu=0, sd=10 * standard_dev_y)
    beta_1 = pm.Normal('beta_1', mu=0, sd=1*standard_dev_y/standard_dev_x1)
    beta_2 = pm.Normal('beta_2', mu=0, sd=1*standard_dev_y/standard_dev_x2)
    epsilon = pm.HalfCauchy('epsilon', 5)
    miu = pm.Deterministic('miu', alfa + beta_1 * x1 + beta_2 * x2)
    y_pred = pm.Normal('y_pred', mu=miu, sd=epsilon, observed=y)

idata_g = pm.sample(400, tune=400, return_inferencedata=True)
az.plot_trace(idata_g, var_names=["α", "β", "ε"])

az.plot_posterior(
         { "beta_1": trace['beta_1'], "beta_2": trace['beta_2']},
         hdi_prob=0.95)
plt.savefig("beta_1beta_2.png")