# -*- coding: utf-8 -*-
"""Untitled35.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nlsNotAsfgjvNKuVG7l6L5uaYyHjbek8
"""

import numpy as np
from scipy.stats import rankdata

def HoeffD(x, y, example=False):
    """
    This will calculate Hoeffding's statistic D.
    Follows section 8.6 of
    Nonparametric Statistical Methods, 3e
    Hollander, Wolfe & Chicken

    Uses the correction for ties given at (8.92).

    It is intended for small sample sizes n only. For large n,
    use the asymptotic equivalence of D to the Blum-Kliefer-Rosenblatt.
    """
    if example:
        x = [7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4]
        y = [2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0]

    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    def phi(a, b):
        if a < b:
            return 1
        elif a == b:
            return 0.5
        else:
            return 0

    c_i = np.zeros(n)
    for i in range(n):
        cc_i = []
        for j in range(n):
            if j != i:
                cc_i.append(phi(x[j], x[i]) * phi(y[j], y[i]))
        c_i[i] = sum(cc_i)

    R_i = rankdata(x)
    S_i = rankdata(y)

    Q = np.sum((R_i - 1) * (R_i - 2) * (S_i - 1) * (S_i - 2))
    R = np.sum((R_i - 2) * (S_i - 2) * c_i)
    S = np.sum(c_i * (c_i - 1))

    D = Q - 2 * (n - 2) * R + (n - 2) * (n - 3) * S
    D /= (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))

    return D