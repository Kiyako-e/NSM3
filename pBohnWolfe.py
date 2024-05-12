import numpy as np
from scipy.stats import beta

def pBohnWolfe(x, y, k, q, c, d, method="Monte Carlo", n_mc=10000):
    outp = {}
    outp["m"] = len(x)
    outp["n"] = len(y)
    outp["n_mc"] = n_mc
    if k * c != outp["m"]:
        print("Warning: Check k*c is the same as the length of x")
    if q * d != outp["n"]:
        print("Warning: Check q*d is the same as the length of y")
    outp["stat_name"] = "Bohn-Wolfe U"

    outp["method"] = method

    if outp["method"] == "Asymptotic":
        print("Warning: The Asymptotic distribution is not yet supported in this version.")
    if outp["method"] == "Exact":
        print("Warning: The Exact distribution is not yet supported in this version.")
    outp["method"] = "Monte Carlo"

    mc_dist = np.zeros(n_mc)

    outp["obs_stat"] = 0
    for j in range(q * d):
        outp["obs_stat"] += np.sum(x < y[j])

    for iter in range(n_mc):
        sample = []
        for j in range(c):
            for i in range(1, k + 1):
                sample.append(beta.rvs(i, k + 1 - i, size=1)[0])
        for j in range(d):
            for i in range(1, q + 1):
                sample.append(beta.rvs(i, q + 1 - i, size=1)[0])
        stat = 0
        for j in range(k * c, k * c + q * d):
            stat += np.sum(sample[:k * c] < sample[j])
        mc_dist[iter] = stat

    mc_vals, mc_probs = np.unique(mc_dist, return_counts=True)
    mc_probs = mc_probs / n_mc

    outp["p_val"] = np.sum(mc_probs[mc_vals >= outp["obs_stat"]])

    return outp
