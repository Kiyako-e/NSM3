# -*- coding: utf-8 -*-
"""Untitled35.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nlsNotAsfgjvNKuVG7l6L5uaYyHjbek8
"""

import numpy as np
from scipy.stats import norm

def cHayStonLSA(alpha, k, delta=0.001):
    our_grid = np.arange(-8, 8, delta)
    len_grid = len(our_grid)

    cutoffs = our_grid[::-1]
    for iter in range(len_grid):
        init_grid = norm.cdf(our_grid + cutoffs[iter])

        if k > 2:
            for i in range(3, k+1):
                new_grid = np.cumsum(init_grid * norm.pdf(our_grid) * delta) + \
                           init_grid * (norm.cdf(our_grid + cutoffs[iter]) - norm.cdf(our_grid))
                init_grid = new_grid

        if np.sum(norm.pdf(our_grid) * init_grid * delta) <= (1 - alpha):
            return cutoffs[iter]