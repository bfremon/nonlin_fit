#!/usr/bin/python3.7

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
vals_nb = 60

np.random.seed(seed=123)

df = {'x': [], 'y': []}

df['x'] = scipy.stats.norm.rvs(81.1, 0.1, vals_nb).tolist() \
    + scipy.stats.norm.rvs(81.5, 0.1, vals_nb).tolist() \
    + scipy.stats.norm.rvs(81.9, 0.1, vals_nb).tolist()
df['y'] = scipy.stats.norm.rvs(18, 2, vals_nb).tolist() \
    + scipy.stats.norm.rvs(8, 3,  vals_nb).tolist() \
    + scipy.stats.norm.rvs(7, 2.5, vals_nb).tolist()

dat = pd.DataFrame(df)
# sns.scatterplot(x=dat['x'], y=dat['y'])
# plt.show()

dat.to_csv('test.csv', sep=';')
