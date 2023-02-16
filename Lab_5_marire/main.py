import pymc3 as pm

# Definim modelul nostru
with pm.Model() as model:
    # Definim variabilele aleatoare
    w = pm.Bernoulli('w', 0.5)  # probabilitatea că prognoza meteo este corectă
    b = pm.Bernoulli('b', 0.8)  # probabilitatea că citirea barometrului este corectă
    p = pm.Deterministic('p', w * b)  # probabilitatea că va ploua

    # Observăm citirea barometrului scăzut
    b_observed = pm.Bernoulli('b_observed', 0.1, observed=True)

    # Obținem probabilitatea condiționată a lui P
    p_observed = pm.Bernoulli('p_observed', p, observed=True)

    # Realizăm inferența
    trace = pm.sample(10000)

pm.plot_posterior(trace['p'])

# Definim modelul
with pm.Model() as model:
    # Definim variabilele aleatoare
    weather = pm.Categorical('weather', [0.3, 0.4, 0.3],
                             labels=['sunny', 'cloudy', 'rainy'])
    rain = pm.Bernoulli('rain',
                        p={('sunny',): 0.1, ('cloudy',): 0.7, ('rainy',): 0.9},
                        shape=1)
    barometer = pm.Bernoulli('barometer',
                             p={True: {True: 0.9, False: 0.2},
                                False: {True: 0.1, False: 0.8}},
                             shape=1)

    # Definim relațiile dintre variabilele aleatoare
    indicator = pm.math.switch(barometer, 1 - rain, rain)
    observed_rain = pm.Deterministic('observed_rain', indicator)

    # Definim observațiile
    rain_observed = pm.Bernoulli('rain_observed',
                                 p=observed_rain,
                                 observed=1)

    # Inferăm distribuțiile posterioare
    trace = pm.sample(5000, tune=1000, random_seed=123)

# Calculăm probabilitatea condiționată P(weather|rain=t)
prob = (trace['weather'] == 1).sum() / len(trace)
print(f"Probabilitatea condiționată P(weather|rain=t) = {prob:.2f}")

