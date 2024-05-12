import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

def approx_ks_d(n):
    """
    Approximations for the critical level for Kolmogorov-Smirnov
    statistic D, for confidence level 0.95.
    Taken from Bickel & Doksum, table IX, p.483
    and Lienert G.A.(1975) who attributes to Miller,L.H.(1956), JASA
    """
    if n > 80:
        return 1.358 / (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))
    else:
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]
        y = [0.975, 0.84189, 0.7076, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925, 0.3376, 0.29408, 0.2417, 0.21012, 0.18841, 0.17231, 0.15975, 0.14960]
        interp_func = interp1d(x, y, kind='cubic')
        return interp_func(n)

def ecdf_ks_ci(x, main=None, sub=None, xlab=None, **kwargs):
    """
    Empirical cumulative distribution function (ECDF) with 95% Kolmogorov-Smirnov confidence bands.

    Parameters:
    x (array-like): Input data.
    main (str, optional): Main title for the plot.
    sub (str, optional): Subtitle for the plot.
    xlab (str, optional): Label for the x-axis.
    **kwargs: Additional arguments passed to the plot function.

    Returns:
    dict: A dictionary containing the lower and upper confidence bands.
    """
    n = len(x)

    if main is None:
        main = f"ESDF + 95% K.S.bands"
    if sub is None:
        sub = f"n = {n}"

    x_sorted = np.sort(x)
    y = np.arange(1, n + 1) / n

    d = approx_ks_d(n)

    y_upper = np.minimum(y + d, 1)
    y_lower = np.maximum(y - d, 0)

    F_X = ECDF(x_sorted)

    plt.figure(figsize=(8, 6))
    plt.step(x_sorted, F_X(x_sorted), label="ECDF", **kwargs)
    plt.step(x_sorted, y_upper, 'r--', label="Upper 95% K.S. band", **kwargs)
    plt.step(x_sorted, y_lower, 'r--', label="Lower 95% K.S. band", **kwargs)
    plt.suptitle(main)
    plt.title(sub)
    plt.xlabel(xlab)
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "lower": y_lower,
        "upper": y_upper
    }
