#!/usr/bin/python3

import pandas as pd
import numpy as np
import nlin_fit as nlf

def loglogistic(x, c1, c2, c3, c4):
        ret = c1 + (c2 - c1) / (1 + np.exp(c4 * np.log(x / c3)))
        return ret

loglog_bounds = {'c3': {'min': 0.0001,
                        'max': np.Inf}
}

loglogparams = {'c1': 1.0,
                'c2': 2.0,
                'c3': 3.0,
                'c4': 4.0}

# Normalize data
dat = pd.read_csv('test.csv', sep=';')
x = nlf.normalize(dat['x'])
y = nlf.normalize(dat['y'])

# Best data fit

fit_res = nlf.fit(loglogistic, x, y, loglog_bounds, **loglogparams)

# Confidence interval

ci_params = nlf.bs_fit_params(loglogistic, x, y, 
                              bounds=loglog_bounds, bs_nb=10**2,
                              **loglogparams)
ci_l, ci_h = nlf.bs_fit_ci(loglogistic, x, ci_params)

# Prediction interval calculations

preds, pi_l, pi_h = nlf.mc_fit_pi(loglogistic, x, y, fit_res,
                              10**3, ret_preds=True)
# Plotting

fname='test'
nlf.plt_fit_ci(dat, fit_res, ci_l, ci_h, pi_l, pi_h, fname=fname)
nlf.plt_residuals(dat, fit_res, fname=fname)

