#!/usr/bin/python3.7

import pandas as pd
import numpy as np
import scipy
import lmfit

def normalize(v):
    ''' Normalize v on [0, 1]'''
    ret = None
    min_v = np.min(v)
    max_v = np.max(v)
    ret = (v - min_v) / (max_v - min_v)
    return ret 

if __name__ == '__main__':
    import unittest

    class test_normalize(unittest.TestCase):
        def test_normalize(self):
            x = scipy.stats.norm.rvs(10, 3, 100)
            r = normalize(x)
            self.assertTrue(np.min(r) == 0.0)
            self.assertTrue(np.max(r) == 1.0)
            self.assertTrue(len(r) == len(x))

    unittest.main()
