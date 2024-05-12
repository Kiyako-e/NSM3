import math
from scipy.special import binom
from scipy.stats import norm, expon

def kolmogorov(x, fnc=norm.cdf, **kwargs):
    def find_kol(d, n):
        sum_val = 0
        sum_max = int(n * (1 - d))
        for j in range(sum_max + 1):
            prod_1 = binom(n, j)
            prod_2 = (1 - d - (j / n)) ** (n - j)
            prod_3 = (d + (j / n)) ** (j - 1)
            sum_val += prod_1 * prod_2 * prod_3
        return 2 * d * sum_val

    def f_n(data, x):
        # F_{n}(x) returns # of X's in the sample(data) <= x / n
        return len(data[data <= x]) / len(data)

    x = sorted(x)
    x_unique = sorted(set(x))

    if len(x_unique) == len(x):  # there are no ties
        D = 0
        n = len(x)
        f_0 = fnc(x, **kwargs)
        for i in range(n):
            D = max(D, abs(((i+1) / n) - f_0[i]), abs(((i) / n) - f_0[i]))
    else:
        # there are ties
        fn = []
        f_0 = fnc(x_unique, **kwargs)
        for j in x_unique:
            fn.append(f_n(x, j))
        D = abs(fn[0] - f_0[0])  # i=1
        for i in range(1, len(x_unique)):
            D = max(D, abs(fn[i] - f_0[i]), abs(fn[i - 1] - f_0[i]))

    p = find_kol(D, n)
    print(f"D= {D}\np= {p}")
    return {"D": D, "p": p}
