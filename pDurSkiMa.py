import numpy as np
from math import factorial
from itertools import permutations
# import multCh7


def pDurSkiMa(x, b=None, trt=None, method=None, n_mc=10000):
    outp = {}
    outp["stat_name"] = "Durbin, Skillings-Mack D"
    outp["n_mc"] = n_mc

    ties = len(np.unique(np.array(x, dtype=float))) != len(x)

    if isinstance(x, np.ndarray) and len(x.shape) == 2:
        outp["n"] = n = x.shape[0]
        outp["k"] = k = x.shape[1]
    else:
        if (len(x) != len(b)) or (len(x) != len(trt)):
            raise ValueError("'x', 'b', and 'trt' must have the same length")

        outp["n"] = n = len(np.unique(b))
        outp["k"] = k = len(np.unique(trt))
        x_vec = x
        num_obs = len(x_vec)
        b_ind = np.array([int(f) for f in np.unique(b, return_inverse=True)[1]]) + 1
        trt_ind = np.array([int(f) for f in np.unique(trt, return_inverse=True)[1]]) + 1

        x = np.zeros((outp["n"], outp["k"]))
        x[:] = np.nan
        for i in range(num_obs):
            x[b_ind[i] - 1, trt_ind[i] - 1] = x_vec[i]

    if len(np.unique(np.sum(~np.isnan(x), axis=1))) != 1:
        raise ValueError("Must be same number of observations per block")
    if len(np.unique(np.sum(~np.isnan(x), axis=0))) != 1:
        raise ValueError("Must be same number of observations per treatment")

    outp["ss"] = s = np.sum(~np.isnan(x[0]))
    outp["pp"] = p = np.sum(~np.isnan(x[:, 0]))
    outp["lambda"] = outp["pp"] * (outp["ss"] - 1) / (outp["k"] - 1)

    outp["obs_mat"] = np.where(~np.isnan(x), 1, 0)
    outp["x"] = x

    if method is None:
        if factorial(outp["ss"]) ** outp["n"] <= 10000:
            method = "Exact"
        else:
            method = "Monte Carlo"

    outp["method"] = method

    possible_ranks = np.apply_along_axis(lambda y: rankdata(y), 1, x[~np.isnan(x)].reshape((-1, outp["ss"])))

    def DSK_stat(obs_data):
        tmp_mat = outp["obs_mat"].copy()
        for i in range(outp["n"]):
            tmp_mat[i, tmp_mat[i] != 0] = obs_data[i]
        Rj = np.sum(tmp_mat, axis=0)
        D_stat = 12 / (outp["lambda"] * outp["k"] * (outp["ss"] + 1)) * np.sum((Rj - outp["pp"] * (outp["ss"] + 1) / 2) ** 2)
        return D_stat

    outp["obs_stat"] = DSK_stat(possible_ranks)

    if outp["method"] == "Exact":
        possible_perm = multCh7(possible_ranks)
        exact_dist = np.apply_along_axis(DSK_stat, axis=1, arr=possible_perm)
        outp["p_val"] = np.mean(exact_dist >= outp["obs_stat"])
    elif outp["method"] == "Monte Carlo":
        mc_perm = np.zeros((outp["n"], outp["ss"]))
        mc_stats = np.zeros(outp["n_mc"])
        for i in range(outp["n_mc"]):
            for j in range(outp["n"]):
                mc_perm[j] = possible_ranks[j, np.random.permutation(outp["ss"])]
            mc_stats[i] = DSK_stat(mc_perm)
        outp["p_val"] = np.mean(mc_stats >= outp["obs_stat"])
    elif outp["method"] == "Asymptotic":
        from scipy.stats import chi2
        outp["p_val"] = 1 - chi2.cdf(outp["obs_stat"], outp["k"] - 1)

    return outp
