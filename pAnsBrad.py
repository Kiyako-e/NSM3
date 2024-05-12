import numpy as np
from scipy.stats import ansari
from math import comb
from itertools import combinations

def pAnsBrad(x, y=None, g=None, method=None, n_mc=10000):
    outp = {}

    if isinstance(x, list):
        if len(x) < 2:
            raise ValueError("'x' must be a list with at least 2 elements")
        y = x[1]
        x = x[0]
    else:
        if np.any(np.isnan(y)):
            k = len(np.unique(g))
            if len(x) != len(g):
                raise ValueError("'x' and 'g' must have the same length")
            if k < 2:
                raise ValueError("all observations are in the same group")
            y = x[g == 1]
            x = x[g == 0]

    outp["m"] = len(x)
    outp["n"] = len(y)
    outp["ties"] = (len(np.concatenate((x, y))) != len(np.unique(np.concatenate((x, y)))))
    outp["extra"] = None
    even = (outp["m"] + outp["n"] + 1) % 2

    outp["stat_name"] = "Ansari-Bradley C"

    if method is None:
        if outp["ties"]:
            if comb(outp["m"] + outp["n"], outp["n"]) <= 10000:
                method = "Exact"
            else:
                method = "Monte Carlo"
        else:
            if outp["m"] + outp["n"] <= 200:
                method = "Exact"
            else:
                method = "Asymptotic"

    outp["method"] = method

    if not outp["ties"]:
        if outp["method"] == "Monte Carlo":
            print("The exact computation will work for large data without ties, so Exact methods are used rather than Monte Carlo.")
            outp["method"] = "Exact"

        if outp["method"] == "Exact":
            tmp = ansari(y, x, alternative="less", exact=True)
            tmp2 = ansari(y, x, exact=True)
        elif outp["method"] == "Asymptotic":
            tmp = ansari(y, x, alternative="less", exact=False)
            tmp2 = ansari(y, x, exact=False)

        outp["obs_stat"] = float(tmp.statistic)
        outp["p_val"] = tmp.pvalue
        outp["two_sided"] = tmp2.pvalue
    else:
        if outp["method"] != "Asymptotic":
            our_data = np.vstack((np.concatenate((x, y)), np.concatenate((np.ones(len(x)), np.zeros(len(y))))))
            sorted_data = our_data[0, np.argsort(our_data[0])]
            x_labels = our_data[1, np.argsort(our_data[0])]

            N = len(sorted_data)
            med = (N + 1) // 2
            if N % 2 == 0:
                no_ties = np.concatenate((np.arange(1, med + 1), np.arange(med, 0, -1)))
            else:
                no_ties = np.concatenate((np.arange(1, med + 1), np.arange(med - 1, 0, -1)))

            obs_group = np.zeros(N, dtype=int)
            group_num = 1

            for i in range(N):
                if obs_group[i] == 0:
                    obs_group[i] = group_num
                    for j in range(i, N):
                        if sorted_data[i] == sorted_data[j]:
                            obs_group[j] = obs_group[i]
                    group_num += 1

            group_ranks = [np.mean(no_ties[obs_group == i]) for i in np.unique(obs_group)]
            tied_ranks = np.zeros(N)
            for i, rank in enumerate(group_ranks):
                tied_ranks[obs_group == i+1] = rank

            assigned_scores = tied_ranks
            outp["obs_stat"] = sum(tied_ranks[x_labels == 0])

            if outp["method"] == "Exact":
                possible_orders = list(combinations(range(outp["m"] + outp["n"]), outp["n"]))
                C_stats = [sum(assigned_scores[list(order)]) for order in possible_orders]
                C_tab = np.unique(C_stats, return_counts=True)
                C_vals = np.round(C_tab[0], 5)
                C_probs = C_tab[1] / np.sum(C_tab[1])
                outp["p_val"] = np.sum(C_probs[C_vals >= np.round(outp["obs_stat"], 5)])
                outp["two_sided"] = 2 * min(outp["p_val"], 1 - outp["p_val"])
            elif outp["method"] == "Monte Carlo":
                outp["n_mc"] = n_mc
                outp["p_val"] = 0
                for _ in range(n_mc):
                    if sum(np.random.choice(assigned_scores, outp["n"], replace=False)) >= outp["obs_stat"]:
                        outp["p_val"] += 1 / n_mc
                outp["two_sided"] = 2 * min(outp["p_val"], 1 - outp["p_val"])
            elif outp["method"] == "Asymptotic":
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmp = ansari(y, x, alternative="greater", exact=False)
                    tmp2 = ansari(y, x, exact=False)
                outp["obs_stat"] = float(tmp.statistic)
                outp["p_val"] = tmp.pvalue
                outp["two_sided"] = tmp2.pvalue

            if not even and outp["method"] == "Exact":
                outp["extra"] = "(N is odd so the null distribution is not symmetric and so the two-sided p-value is approximate.)"

            return outp
