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
def min_setup():
    dat = pd.read_csv('test.csv', sep=';')
    thres = nlf.normalize_val(dat['y'], 3)
    nlf.fit_fun(loglogistic, dat, 'loglogistic', thres=thres,
            bounds=loglog_bounds, **loglogparams)

min_setup()
