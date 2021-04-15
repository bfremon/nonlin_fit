#!/usr/bin/python3.7

import os
import pandas as pd
import numpy as np
import scipy
import lmfit
import matplotlib.pyplot as plt
import seaborn as sns
import time

def normalize(v):
    ''' Normalize v on [0, 1]'''
    ret = None
    min_v = np.min(v)
    max_v = np.max(v)
    ret = (v - min_v) / (max_v - min_v)
    return ret 


def sigmoid_2p(x, c1, c2):
    ret = 1 / (1 + np.exp(c1 * (x - c2)))
    return ret


def _parse_bounds(bounds, prms, params):
    for k in bounds:
        if not k in prms:
            raise SyntaxError('%s not a func arg' % k)
        for b in bounds[k]:
            if b != 'min' or b != 'max':
                raise SyntaxError('Only max or min allowed as bound keywords')
            if b == 'min':
                params[k].min = bound[k]['min']
            else:
                params[k].max = bound[k]['max']

                
def fit(func, x, y, bounds=None, **kwargs):
    '''
    Fit y = func() using params using lmfit.
    y: dependent variable
    x: independent variable
    **kwargs holds minimum value for each parameter of func 
    bounds optionally holds min / max values for parameters of func 
    return result object
    '''
    mod = lmfit.Model(func) #, independent_vars = ['x'])
    params = mod.make_params()
    for k in kwargs:
        params.add(k, kwargs[k])
    if bounds:
        _parse_bounds(bounds, kwargs, params)
    ret = mod.fit(y, params, x=x)
    return ret


def resample(x, y, vals_nb):
    '''
    Resample without replacement vals_nb in x and y
    '''
    if len(x) != len(y):
        raise ValueError('len(x) must be equal to len(y)')
    if vals_nb > len(x):
        raise ValueError('len(x) must be inferior to vals_nb')
    ret = {'x': [], 'y': []}
    for i in range(vals_nb):
        idx = int(len(x) * np.random.uniform())
        ret['x'].append(x[idx])
        ret['y'].append(y[idx])
    return ret


def bs_fit_params(func, x, y, bounds=None, bs_nb=100, **kwargs):
    fits_data = {}
    ret = {}
    for i in range(bs_nb):
        resamples = resample(x, y, len(x))
        res = fit(func, resamples['x'], resamples['y'], bounds, **kwargs)
        fits_data[i] = {'x': resamples['x'],
                  'y': resamples['y'],
                  'fit': res}
    for f in fits_data:
        ret[f] = {}
        for k in fits_data[f]['fit'].params:
          ret[f][k] = fits_data[f]['fit'].params[k].value
    return ret

def bs_fit_ci(func, x, params, alpha=0.05):
    ci_data = {}
    for k in params:
        ci_data[k] = func(x, **params[k])
    ci_df = pd.DataFrame(ci_data)
    lb = float(alpha / 2)
    ub = 1 - lb
    low = ci_df.quantile(lb, axis='columns')
    high = ci_df.quantile(ub, axis='columns')
    return low, high

def bs_fit_pi(func, x, y, fit_res, draws_nb=10**4, alpha=0.05):
    ypred = func(x, **fit_res.params)
    noise = np.std(y - ypred)
    preds = np.array([np.random.normal(ypred, noise) for j in range(draws_nb)])
    l = float(alpha / 2)
    u = 1 - l
    low = np.quantile(preds, l, axis=0)
    high = np.quantile(preds, u, axis=0)
    return low, high
                     
                     
def plt_bs_fit(fits, out=os.getcwd()):
    sns.scatterplot(x=x, y=y, color='orange')
    for f in fits:
        xobs = fits[f]['x']
        yobs = fits[f]['y']
        ypred = fits[f]['fit'].best_fit
        lab = str(fits[f]['fit'].aic) + ' '  + str(fits[f]['fit'].bic)
        sns.scatterplot(x=xobs, y=yobs, color='orange')
        sns.lineplot(x=xobs, y=ypred, label=lab)
        out_f = os.path.join(out, 'tmp',  str(f) + '.png')
        plt.savefig(out_f)
        plt.cla()

def plt_fit(x, y, fit_res):
    '''
    Plot fit results from fit_res (output of oneDfit()
    '''
    residuals = y - fit_res.best_fit
    uncert = fit_res.eval_uncertainty(fit_res.params, sigma=2)
    sns.scatterplot(x=x, y=y, label='exp', color='orange')
    sns.lineplot(x=x, y=fit_res.best_fit, label='fit', color='lightgreen')
    sns.lineplot(x=x, y=fit_res.best_fit - uncert, label='fit CI',
                    color='lightblue')
    sns.lineplot(x=x, y=fit_res.best_fit + uncert, color='lightblue')
    plt.show()

def plt_fit_ci(x, y, fit_res, ci_l, ci_h, pi_l, pi_h, out_d=os.getcwd(), fname=None):
        sns.scatterplot(x=x, y=y, color='orange')
        sns.lineplot(x=x, y=fit_res.best_fit, label='fit', color='blue')
        sns.lineplot(x=x, y=ci_l, label='fit CI', color='lightblue')
        sns.lineplot(x=x, y=ci_h, color='lightblue')
        sns.lineplot(x=x, y=pi_l, color='lightgreen', label='fit PI')
        sns.lineplot(x=x, y=pi_h, color='lightgreen')
        if fname:
            out_f = os.path.join(out_d, str(fname) + '.png')
        else:
            fn = time.time().split('.')[0]
            out_f = os.path.join(out_d, fn)
        plt.savefig(out_f, dpi=600)
        plt.cla()
    
if __name__ == '__main__':

    def test():
        dat = pd.read_csv('test.csv', sep=';')
        x = normalize(dat['x'])
        y = normalize(dat['y'])
        
        fit_res = fit(sigmoid_2p, x, y, c1=2.0, c2=1.0)

        ci_params = bs_fit_params(sigmoid_2p, x, y, c1=2.0, c2=1.0, bs_nb=1000)
        ci_l, ci_h = bs_fit_ci(sigmoid_2p, x, ci_params)
        pi_l, pi_h = bs_fit_pi(sigmoid_2p, x, y, fit_res, 10**4)
        plt_fit_ci(x, y, fit_res, ci_l, ci_h, pi_l, pi_h, fname='2p-sigmoid')

    test()
    
    # import unittest

    # class test_normalize(unittest.TestCase):
        
    #     def test_normalize(self):
    #         x = scipy.stats.norm.rvs(10, 3, 100)
    #         r = normalize(x)
    #         self.assertTrue(np.min(r) == 0.0)
    #         self.assertTrue(np.max(r) == 1.0)
    #         self.assertTrue(len(r) == len(x))

    #     def _sigmoid_2p(x, c1, c2):
    #         ret = 1 / (1 + np.exp(c1 * (x - c2)))
    #         return ret

    #     def test_fit(self):
    #         x = scipy.stats.norm.rvs(0, 1, 100)
    #         ytest = scipy.stats.norm.rvs(0, 1, 101)
    #         self.assertRaises(SyntaxError, fit, self._sigmoid_2p, \
    #                           x, ytest, c1=1.0, c2=2.0)
    #    unittest.main()










