import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

#Ex1
centered_posterior_data = centered_data['posterior']

print(centered_posterior_data)

graph= az.plot_trace(centered_data, divergences='top', compact=False)

non_centered_posterior_data = non_centered_data['posterior']

print(non_centered_posterior_data)

graph = az.plot_trace(non_centered_data, divergences='top', compact=False)


#Ex2
rhat_value1 = az.rhat(centered_data, var_names=["mu", "theta"])
rhat_value2 = az.rhat(non_centered_data, var_names=["mu", "theta"])

print(rhat_value1)
print(rhat_value2)


#Ex3
centered_data.sample_stats.diverging.sum()
non_centered_data.sample_stats.diverging.sum()

_, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 10), constrained_layout=True)

for index, tr in enumerate([centered_data, non_centered_data]):
    az.plot_pair(tr, var_names=['mu', 'tau'], kind='scatter',
                 divergences=True, divergences_kwargs={'color':'C1'},
                 ax=ax[index])

    ax[index].set_title(['centered', 'non-centered'][index])

plt.show()