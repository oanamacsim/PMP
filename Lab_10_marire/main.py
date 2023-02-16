import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Load iris dataset
iris = pd.read_csv('iris.csv')

# Select sepal length as the feature
X = iris['sepal_length'].values.reshape(-1, 1)

with pm.Model() as model:
    # Priors for intercept and coefficients
    beta0 = pm.Normal('beta0', mu=0, sd=100)
    beta1 = pm.Normal('beta1', mu=0, sd=100)

    # Priors for variances
    s_0 = pm.Gamma('s_0', alpha=0.001, beta=0.001)
    s_1 = pm.Gamma('s_1', alpha=0.001, beta=0.001)
    s_2 = pm.Gamma('s_2', alpha=0.001, beta=0.001)

    # Indicator variable for each species
    species = pm.Categorical('species', p=[0.33, 0.33, 0.33], shape=len(X))

    # Expected value
    mu = beta0 + beta1 * X[species]

    # Likelihood for each observation
    obs = pm.Normal('obs', mu=mu, sd=pm.math.sqrt(s_0 + s_1 * (species == 0) + s_2 * (species == 1)), observed=X)

with model:
    trace = pm.sample(1000, tune=1000, cores=1)

pm.traceplot(trace)

colors = {'setosa':'r', 'versicolor':'g', 'virginica':'b'}
for i in range(len(X)):
    plt.scatter(X[i], 0, c=colors[iris['species'][i]])
plt.show()
