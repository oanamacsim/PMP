import numpy as np
import pymc3 as pm

# Definirea datelor
x = np.array([4., 5., 6., 9., 12, 14.])
y = np.array([4.2, 6., 6., 9., 10, 10.])

# Definirea funcției de regresie
def poly_func(x, coeffs):
    return np.polyval(coeffs, x)

# Definirea modelelor pentru fiecare grad de polinom
with pm.Model() as model0:
    # Polinom de grad 0
    y_obs = pm.Normal("y_obs", mu=y.mean(), sd=y.std(), observed=y)
    trace0 = pm.sample(1000, tune=1000)
    waic0 = pm.waic(trace0, model0)
    loo0 = pm.loo(trace0, model0)

with pm.Model() as model1:
    # Polinom de grad 1
    coeffs1 = pm.Normal("coeffs1", mu=0, sd=10, shape=2)
    y_obs = pm.Normal("y_obs", mu=poly_func(x, coeffs1), sd=y.std(), observed=y)
    trace1 = pm.sample(1000, tune=1000)
    waic1 = pm.waic(trace1, model1)
    loo1 = pm.loo(trace1, model1)

with pm.Model() as model2:
    # Polinom de grad 2
    coeffs2 = pm.Normal("coeffs2", mu=0, sd=10, shape=3)
    y_obs = pm.Normal("y_obs", mu=poly_func(x, coeffs2), sd=y.std(), observed=y)
    trace2 = pm.sample(1000, tune=1000)
    waic2 = pm.waic(trace2, model2)
    loo2 = pm.loo(trace2, model2)

with pm.Model() as model5:
    # Polinom de grad 5
    coeffs5 = pm.Normal("coeffs5", mu=0, sd=10, shape=6)
    y_obs = pm.Normal("y_obs", mu=poly_func(x, coeffs5), sd=y.std(), observed=y)
    trace5 = pm.sample(1000, tune=1000)
    waic5 = pm.waic(trace5, model5)
    loo5 = pm.loo(trace5, model5)

print(waic0.waic)
print(waic1.waic)
print(waic2.waic)
print(waic5.waic)
print(loo0.loo)
print(loo1.loo)
print(loo2.loo)
print(loo5.loo)

