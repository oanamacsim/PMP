import pandas as pd
import pymc3 as pm


df = pd.read_csv('Titanic.csv')
# Gestionarea valorilor lipsă
df = df.dropna(subset=['Pclass', 'Age', 'Survived'])

# Transformarea variabilelor
df['Pclass'] = df['Pclass'].astype('category')
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
df['Survived'] = df['Survived'].astype('category').cat.codes

with pm.Model() as model:
    # Coeficienții pentru Pclass și Age
    β0 = pm.Normal('β0', mu=0, sd=10)
    β1 = pm.Normal('β1', mu=0, sd=10)
    β2 = pm.Normal('β2', mu=0, sd=10)

    # Probabilitatea pentru fiecare pasager de a supraviețui
    p = pm.math.invlogit(β0 + β1 * df['Pclass'].cat.codes.values + β2 * df['Age'].values)

    # Variabila de supraviețuire (0 = nu a supraviețuit, 1 = a supraviețuit)
    observed = pm.Bernoulli('observed', p=p, observed=df['Survived'].values)

with model:
    trace = pm.sample(5000)

pm.summary(trace)

decision_boundary = -(β0 + β1 * Pclass + β2 * Age) / β2
import numpy as np
import matplotlib.pyplot as plt

# Valorile pentru Pclass și Age
x1 = np.array([1, 2, 3])
x2 = np.linspace(df['Age'].min(), df['Age'].max(), 100)

# Distribuția posterioră pentru β0, β1 și β2
β0_post = trace['β0']
β1_post = trace['β1']
β2_post = trace['β2']

# Valorile medii ale coeficienților
β0_mean = np.mean(β0_post)
β1_mean = np.mean(β1_post)
β2_mean = np.mean(β2_post)

# Calculul graniței de decizie
decision_boundary = -(β0_mean + β1_mean * x1[:,None] + β2_mean * x2) / β2_mean

# Calculul intervalului HDI de 95% pentru coeficienții β0, β1 și β2
hdi = pm.stats.hdi(trace['β0'], hdi_prob=0.95), pm.stats.hdi(trace['β1'], hdi_prob=0.95), pm.stats.hdi(trace['β2'], hdi_prob=0.95)

# Graficul graniței de decizie și a intervalului HDI de 95%
plt.plot(x2, decision_boundary.T, label='Decision boundary')
plt.fill_between(x2, pm.math.invlogit(-(hdi[0][1] + hdi[1][1] * x1[:,None] + hdi[2][1] * x2)) / hdi[2][1],
                 pm.math.invlogit(-(hdi[0][0] + hdi[1][0] * x1[:,None] + hdi[2][0] * x2)) / hdi[2][0], alpha=0.3, label='95% HDI')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Pclass')
plt.show()
