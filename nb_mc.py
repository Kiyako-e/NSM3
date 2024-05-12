import numpy as np
from scipy.stats import norm

def find_nbu(x):
    n = len(x)
    y = np.sort(x)
    a = np.zeros((n, n, n))

    for i in range(2, n):
        for j in range(1, i):
            for k in range(j):
                if y[i] > y[j] + y[k]:
                    a[i, j, k] = 1
                elif y[i] == y[j] + y[k]:
                    a[i, j, k] = 0.5
                else:
                    a[i, j, k] = 0

    return np.sum(a)

def p_g_mc(T, n, min_reps=100, max_reps=1000, delta=1e-3):
    dsn = []
    for _ in range(min_reps):
        dsn.append(find_nbu(np.random.exponential(1, n)))

    reps = min_reps
    while reps <= max_reps:
        p = len([x for x in dsn if x > T]) / reps
        dsn.append(find_nbu(np.random.exponential(1, n)))
        if abs(p - len([x for x in dsn if x > T]) / reps) <= delta:
            return p
        reps += 1

    print("Warning: reached maximum reps without converging within delta")
    return p

def p_l_mc(T, n, min_reps=100, max_reps=1000, delta=1e-3):
    dsn = []
    for _ in range(min_reps):
        dsn.append(find_nbu(np.random.exponential(1, n)))

    reps = min_reps
    while reps <= max_reps:
        p = len([x for x in dsn if x < T]) / reps
        dsn.append(find_nbu(np.random.exponential(1, n)))
        if abs(p - len([x for x in dsn if x < T]) / reps) <= delta:
            return p
        reps += 1

    print("Warning: reached maximum reps without converging within delta")
    return p

def nb_mc(x, alternative="two.sided", exact=False, min_reps=100, max_reps=1000, delta=1e-3):
    def char_expand(alternative, options):
        if alternative in options:
            return alternative
        else:
            raise ValueError("alternative must be one of: {}".format(", ".join(options)))

    alternative = char_expand(alternative, ["two.sided", "nbu", "nwu"])

    T = find_nbu(x)
    n = len(x)

    if n >= 9 and not exact:
        b = T
        e = n * (n - 1) * (n - 2) / 8
        g = (3 / 2) * n * (n - 1) * (n - 2)
        h = (5 / 2592) * (n - 3) * (n - 4)
        i = (n - 3) * (7 / 432)
        j = 1 / 48
        k = g * (h + i + j)
        s = np.sqrt(k)
        T_star = (b - e) / s

        if alternative == "nbu":
            p = norm.cdf(T_star)
        elif alternative == "nwu":
            p = 1 - norm.cdf(T_star)
        else:
            p = 2 * (1 - norm.cdf(abs(T_star)))

        print(f"T*= {T_star:.2f}\np= {p:.4f}")
        return {"T": T_star, "prob": p}

    if alternative == "nbu":
        p = p_l_mc(T, n, min_reps, max_reps, delta)
    elif alternative == "nwu":
        p = p_g_mc(T, n, min_reps, max_reps, delta)
    else:
        p_l = p_l_mc(T, n, min_reps, max_reps, delta)
        p_g = p_g_mc(T, n, min_reps, max_reps, delta)
        p = 2 * min(p_l, p_g)

    print(f"T= {T}\np= {p:.4f}")
    return {"T": T, "p": p}
