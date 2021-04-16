#!/usr/bin/python3.7

import os
import pandas as pd
import numpy as np
import scipy
import lmfit
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.distributions.empirical_distribution import ECDF

def normalize(v):
    ''' Normalize v on [0, 1]'''
    ret = None
    min_v = np.min(v)
    max_v = np.max(v)
    ret = (v - min_v) / (max_v - min_v)
    return ret 


def _parse_bounds(bounds, prms, params):
    '''
    Private func for bounds parsing
    '''
    for k in bounds:
        if not k in prms:
            raise SyntaxError('%s not a func arg' % k)
        for b in bounds[k]:
            if b == 'min':
                params[k].min = bounds[k]['min']
            elif b == 'max':
                params[k].max = bounds[k]['max']
            else:
                raise SyntaxError('Only max or min allowed as bounds keywords')

            
def fit(func, x, y, bounds=None, **kwargs):
    '''
    Fit y = func() using params using lmfit.
    y: dependent variable
    x: independent variable
    **kwargs holds minimum value for each parameter of func 
    bounds optionally holds min / max values for parameters of func 
    return result object
    '''
    mod = lmfit.Model(func, independent_vars = ['x'])
    params = mod.make_params()
    for k in kwargs:
        params.add(k, value=kwargs[k])
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
    '''
    Bootstrap bs_nb times func(x) parameters 
    func: objective function
    func: objective function
    x: input
    y: func(x) output
    bounds: min / max values for func parameters
    bs_nb: number of bootstraps
    return best_fit params values for each fit  
    '''
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
    '''
    Determine boostrapped confidence intervals for fit parameters
    for func(x)
    func: objective function
    x: input
    y: f(x) output
    params: output of bs_fit_params()
    alpha: confidence level
    Return lower and upper confidence bounds for fit at the alpha level for f(x)
    '''
    ci_data = {}
    for k in params:
        ci_data[k] = func(x, **params[k])
    ci_df = pd.DataFrame(ci_data)
    lb = float(alpha / 2)
    ub = 1 - lb
    low = ci_df.quantile(lb, axis='columns')
    high = ci_df.quantile(ub, axis='columns')
    return low, high


def mc_fit_pi(func, x, y, fit_res, draws_nb=10**4, alpha=0.05, ret_preds=False):
    '''
    Determine predictions intervals at the alpha level using residuals std
    func: objective function
    x: input
    y: f(x) output
    fit_res: fit object holding best_fit params
    draws_nb: number of Monte Carlo draws
    alpha: confidence level
    ret_preds: to return Monte Carlo predictions (option)
    '''
    ypred = func(x, **fit_res.params)
    noise = np.std(y - ypred)
    preds = np.array([np.random.normal(ypred, noise) for j in range(draws_nb)])
    l = float(alpha / 2)
    u = 1 - l
    low = np.quantile(preds, l, axis=0)
    high = np.quantile(preds, u, axis=0)
    if not ret_preds:
        return low, high
    else:
        preds_t = np.transpose(preds)
        preds_df = pd.DataFrame(preds_t)
        return preds_df, low, high
                     
                     
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

    
def _prep_out_f(out_d, fname=None, suffix=None):
    ret = ''
    if fname:
        ret = str(fname)
    else:
        ret = str(time.time()).split('.')[0]
    if suffix:
        ret += suffix
    ret += '.png'
    ret = os.path.join(out_d, ret)
    return ret

def plt_fit_ci(x, y, fit_res, ci_l, ci_h, pi_l, pi_h, out_d=os.getcwd(), fname=None):
        sns.scatterplot(x=x, y=y, color='orange')
        sns.lineplot(x=x, y=fit_res.best_fit, label='fit', color='blue')
        sns.lineplot(x=x, y=ci_l, label='fit CI', color='lightblue')
        sns.lineplot(x=x, y=ci_h, color='lightblue')
        sns.lineplot(x=x, y=pi_l, color='lightgreen', label='fit PI')
        sns.lineplot(x=x, y=pi_h, color='lightgreen')
        out_f = _prep_out_f(out_d, fname)
        plt.savefig(out_f, dpi=600)
        plt.cla()

        
def plt_residuals(x, y, fit_res, out_d=os.getcwd(), fname=None):
    residuals = y - fit_res.best_fit
    sns.scatterplot(x=x, y=residuals)
    out_f = _prep_out_f(out_d, fname, '-residuals')
    plt.savefig(out_f, dpi=300)
    plt.cla()
    scipy.stats.probplot(residuals, dist='norm', plot=plt)
    out_f = _prep_out_f(out_d, fname, '-residuals-probplot')
    plt.savefig(out_f, dpi=300)
    plt.cla()

    
def fit_fun(func, x, y, fname=None, out_d=os.getcwd(),
            bs_nb=100, mc_draws=10**4, thres=None,
            bounds=None, **kwargs):
    print('processing %s' % fname)
    fit_res = fit(func, x, y, bounds=bounds, **kwargs)
    print('main fit done')
    ci_params = bs_fit_params(func, x, y, 
                              bounds=bounds, bs_nb=bs_nb, **kwargs)
    print('CI bootstrap done')
    ci_l, ci_h = bs_fit_ci(func, x, ci_params)
    preds, pi_l, pi_h = mc_fit_pi(func, x, y, fit_res,
                                  mc_draws, ret_preds=True)
    print('PI MC done')
    plt_fit_ci(x, y, fit_res, ci_l, ci_h, pi_l, pi_h, fname=fname)
    plt_residuals(x, y, fit_res, fname=fname)
    if thres != None:
        plt_thres_quantile(x, preds, thres, fname=fname)

def plt_thres_quantile(x, preds, thres, out_d=os.getcwd(), fname=None):
    '''
    For each x prediction in preds (x and preds rows aligned, 
    determine the probability of having a value higher than thres
    x: input
    preds: Monte Carlo predictions (output of mc_fit_pi())
    thres: y-normalized threshold
    '''
    ret = {'x': [], 'y': []}
    i = 0
    for i in range(len(preds)):
        ecdf = ECDF(preds.iloc[i])
        idx = min_idx_val(ecdf.x, thres)
        ret['x'].append(x[i])
        ret['y'].append(ecdf.y[idx])
    yval = 1.0 - np.array(ret['y'])
    xval = np.array(ret['x'])
    sns.lineplot(x=xval, y=yval)
    out_f = _prep_out_f(out_d, fname, '-probablity_over_thres')    
    plt.savefig(out_f, dpi=300)
    plt.cla()

def min_idx_val(x, thres):
    diff = x - float(thres)
    diff[diff < 0.0] = np.Inf
    ret = diff.argmin()
    return ret
    
def normalize_val(x, val):
    '''
    Normalize val from vector x
    '''
    v = np.append(x, val)
    normalized_v = normalize(v)
    ret = normalized_v[len(normalized_v) - 1]
    return ret
    
if __name__ == '__main__':
    def sigmoid_2p(x, c1, c2):
        ret = 1 / (1 + np.exp(c1 * (x - c2)))
        return ret

    def loglogistic(x, c1, c2, c3, c4):
        ret = c1 + ((c2 - c1) / (1 + np.exp(x - c3) / c4))
        return ret

    loglog_bounds = {'c3': {'min': 0.0001,
                            'max': np.Inf}
    }
    loglogparams = {'c1': 1.0,
                    'c2': 2.0,
                    'c3': 3.0,
                    'c4': 4.0}
    def test():
        dat = pd.read_csv('test.csv', sep=';')
        x = normalize(dat['x'])
        y = normalize(dat['y'])
        thres = normalize_val(dat['y'], 3)
        params = {'c1': 1.0, 'c2': 2.0}
        # fit_fun(sigmoid_2p, x, y, 'sigmoid',
        #     thres=thres, bounds=None, **params)
        fit_fun(loglogistic, x, y, 'loglogistic', thres=thres,
                bounds=loglog_bounds, **loglogparams)
    test()
    
    import unittest

    class test_normalize(unittest.TestCase):
        def test_min_idx_val(self):
            v = np.array([ float(i) for i in range(30)])
            r = min_idx_val(v, 3)
            self.assertTrue(r == 3)
            r = min_idx_val(v, 31.0)
            self.assertTrue(r == 0)
            v = np.array([float(i) for i in range(-3, 10)])
            r = min_idx_val(v, 3)
            print(r)
                         
        def test_normalize(self):
            x = scipy.stats.norm.rvs(10, 3, 100)
            r = normalize(x)
            self.assertTrue(np.min(r) == 0.0)
            self.assertTrue(np.max(r) == 1.0)
            self.assertTrue(len(r) == len(x))

        def _sigmoid_2p(self, x, c1, c2):
            ret = 1 / (1 + np.exp(c1 * (x - c2)))
            return ret

        def _model_init(self, fun, **kwargs):
            mod = lmfit.Model(fun)
            params = mod.make_params()
            for k in kwargs:
                params.add(k, value=params[k])
            return mod, params

        def test__parse_bounds(self):
            fun_args = {'c1': 1.0, 'c2':2.0}
            m1, p1 = self._model_init(self._sigmoid_2p, c1=1.0, c2=2.0)
            bounds = {'c3': {'min': 3.0}}
            self.assertRaises(SyntaxError, _parse_bounds,
                              bounds, fun_args, p1)
            m2, p2 = self._model_init(self._sigmoid_2p, c1=1.0, c2=2.0)
            bounds = {'c1': {'moux': 3.0}}
            self.assertRaises(SyntaxError, _parse_bounds,
                              bounds, fun_args, p2)
            m3, p3 = self._model_init(self._sigmoid_2p, c1=1.0, c2=2.0)
            bounds = {'c1': {'max': 3.0}}
            _parse_bounds(bounds, fun_args, p3)
            v = p3.valuesdict()
            self.assertTrue(v['c1'].min == 3.0)
            m4, p4 = self._model_init(self._sigmoid_2p, c1=1.0, c2=2.0)
            bounds = {'c1': {'min': 3.0}}
            _parse_bounds(bounds, fun_args, p4)
            v = p4.valuesdict()
            self.assertTrue(v['c1'].max == 3.0)
            
        def test_fit(self):
            x = scipy.stats.norm.rvs(0, 1, 100)
            ytest = scipy.stats.norm.rvs(0, 1, 101)
    #        self.assertRaises(SyntaxError, fit, self._sigmoid_2p, \
     #                         x, ytest, c1=1.0, c2=2.0)
    unittest.main()










