import numpy as np
from scipy.stats import norm, rankdata, kendalltau

def kendall_ci(x=None, y=None, alpha=0.05, type="t", bootstrap=False, B=1000, example=False):
    """
    This will produce a 1 - alpha CI for Kendall's tau.
    Based on sections 8.3 and 8.4 of:
    Nonparametric Statistical Methods, 3e
    Hollander, Wolfe & Chicken

    Parameters:
    x (array-like): x sample
    y (array-like): y sample
    alpha (float): significance level (default: 0.05)
    type (str): CI type, can be "t" (two-sided), "l" (lower) or "u" (upper) (default: "t")
    bootstrap (bool): whether to use bootstrap CI (default: False)
    B (int): number of bootstrap replicates (default: 1000)
    example (bool): whether to use the example data from HW&C (default: False)

    Returns:
    None
    """
    # Example 8.1 from HW&C
    if example:
        x = [44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1]
        y = [2.6, 3.1, 2.5, 5.0, 3.6, 4.0, 5.2, 2.8, 3.8]

    continue_flag = True

    if x is None or y is None:
        print("\nYou must supply an x sample and a y sample!\n")
        continue_flag = False

    if continue_flag and len(x) != len(y):
        print("\nSamples must be of the same length!\n")
        continue_flag = False

    if continue_flag and len(x) <= 1:
        print("\nSample size n must be at least two!\n")
        continue_flag = False

    if continue_flag and type not in ["t", "l", "u"]:
        print("\nArgument \"type\" must be one of \"t\" (two-sided), \"l\" (lower) or \"u\" (upper)!\n")
        continue_flag = False

    if continue_flag:
        # Q* from (8.17)
        def Q(i, j):
            ij = (j[1] - i[1]) * (j[0] - i[0])
            if ij > 0:
                return 1
            elif ij < 0:
                return -1
            else:
                return 0

        # C.i from (8.37)
        def C_i(x, y, i):
            c_i = 0
            for k in range(len(x)):
                if k != i:
                    c_i += Q([x[i], y[i]], [x[k], y[k]])
            return c_i

        if not bootstrap:
            c_i = [C_i(x, y, i) for i in range(len(x))]
            tau_hat = kendalltau(x, y).statistic
            sigma_hat_2 = 2 * (len(x) - 2) * np.var(c_i) * len(c_i) / (len(c_i) - 1) / (len(x) * (len(x) - 1))
            sigma_hat_2 += 1 - tau_hat ** 2
            sigma_hat_2 *= 2 / (len(x) * (len(x) - 1))

            if type == "t":
                z = norm.ppf(1 - alpha / 2)
            else:
                z = norm.ppf(1 - alpha)

            tau_L = tau_hat - z * np.sqrt(sigma_hat_2)
            tau_U = tau_hat + z * np.sqrt(sigma_hat_2)

            if type == "l":
                tau_U = 1
            elif type == "u":
                tau_L = -1

        if bootstrap:
              tau = []
              for b in range(B):
                  b_sample = np.random.choice(np.arange(0, len(x), 1), len(x))
                  tau_sample = kendalltau(np.array(x)[b_sample], np.array(y)[b_sample]).statistic
                  tau.append(tau_sample)

              tau = sorted(tau)
              if type == "t":
                  k = int(np.floor((B + 1) * alpha / 2))
              else:
                  k = int(np.floor((B + 1) * alpha))

              tau_L = tau[k]
              tau_U = tau[B + 1 - k]

              if type == "l":
                  tau_U = 1
              elif type == "u":
                  tau_L = -1

        tau_L = round(tau_L, 3)
        tau_U = round(tau_U, 3)

        if type == "t":
            print_type = " two-sided CI for tau:"
        elif type == "l":
            print_type = " lower bound for tau:"
        else:
            print_type = " upper bound for tau:"

        print(f"\n1 - alpha = {1 - alpha}{print_type}")
        print(f"{tau_L}, {tau_U}\n")
