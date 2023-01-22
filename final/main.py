from scipy.stats import *
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# A.1.1
poisson_dist = poisson(mu=10)
print(poisson_dist.rvs())
# A.1.2
sample = poisson_dist.rvs(size=100)
plt.hist(sample, bins=20, density=True, alpha=0.6, color='b')
plt.xlabel('Number of Visitors')
plt.ylabel('Probability')
plt.show()

# A.2.1
uniform_dist = uniform(loc=10, scale=20)
print(uniform_dist.rvs())
# A.2.2
sample = uniform_dist.rvs(size=100)
plt.hist(sample, bins=20, density=True, alpha=0.6, color='r')
plt.xlabel('Weight (kg)')
plt.ylabel('Probability')
plt.show()

# A.3.1
norm_dist = norm(loc=5000, scale=500)
print(norm_dist.rvs())
# A.3.2
sample = norm_dist.rvs(size=100)
plt.hist(sample, bins=20, density=True, alpha=0.6, color='g')
plt.xlabel('Weight (kg)')
plt.ylabel('Probability')
plt.show()

# A.4.1
mean, std, skew = 70, 10, 1
distribution = skewnorm(skew, loc=mean, scale=std)
weights = distribution.rvs(size=1000)
# A.4.2
sample = distribution.rvs(size=100)
plt.hist(sample, bins=30, density=True, alpha=0.6, color='g')
plt.xlabel('Weight (kg)')
plt.ylabel('Probability')

plt.show()

# B.1.
Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    theta = pm.Uniform('theta', lower=0, upper=1)
    Y = pm.Binomial('Y', n=n, p=theta, observed=Y_values)
    trace = pm.sample(draws=1000, tune=1000)

pm.plot_posterior(trace, var_names=['n'], color='g', ref_val=10)
plt.show()

#D
# Generate simulated data
np.random.seed(123)
x = np.linspace(-20, 10, 100)
y = 9 + 3 * x + np.random.normal(0, 5, size=100)

# Build linear model in PyMC3
with pm.Model() as model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfStudentT('sigma', nu=3)

    # Expected value of outcome
    mu = alpha + beta * x

    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)

    # Sample posterior
    trace = pm.sample(3000, chains=2)

# Plot posterior distributions
pm.plot_posterior(trace, varnames=['alpha', 'beta', 'sigma'], color='#87ceeb')
plt.show()

# Summary of the posterior distributions
summary = pm.summary(trace)
print(summary)

# Plotting Posterior distributions of the parameters
pm.plot_posterior(trace, varnames=['alpha', 'beta', 'sigma'], credible_interval=0.95, kind='hist')
plt.show()

# Plotting Forest Plot of the parameters
pm.plot_forest(trace, varnames=['alpha', 'beta', 'sigma'], credible_interval=0.95, kind='ridge')
plt.show()

