import numpy as np
from scipy.stats import norm

def newbet(x):
    n = len(x)
    y = np.sort(x)
    a = np.zeros((n, n, n))

    # Compute the NBU statistic
    for i in range(2, n):
        for j in range(1, i):
            for k in range(j):
                if y[i] > y[j] + y[k]:
                    a[i, j, k] = 1
                elif y[i] == y[j] + y[k]:
                    a[i, j, k] = 0.5
                else:
                    a[i, j, k] = 0

    b = np.sum(a)
    e = n * (n - 1) * (n - 2) / 8
    g = (3 / 2) * n * (n - 1) * (n - 2)
    h = (5 / 2592) * (n - 3) * (n - 4)
    i = (n - 3) * (7 / 432)
    j = 1 / 48
    k = g * (h + i + j)
    s = np.sqrt(k)
    p = 1 - norm.cdf(abs((b - e) / s))

    return [b, (b - e) / s, p]
