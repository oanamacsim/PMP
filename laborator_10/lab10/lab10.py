import numpy as np
import arviz as az

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, 2]
std_devs = [2, 2]

mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix));

clusters = [3, 4, 5, 6]
models = []
idatas = []
for cluster in clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means',
        mu=np.linspace(mix.min(), mix.max(), cluster),
        sd=10, shape=cluster,
        transform=pm.distributions.transforms.ordered)

sd = pm.HalfNormal('sd', sd=10)
y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
idatas.append(idata)
models.append(model)

cmp_df = az.compare({idatas},method='BB-pseudo-BMA', ic="waic", scale="deviance")