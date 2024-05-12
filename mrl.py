import numpy as np
import matplotlib.pyplot as plt

def mrl(data, alpha, main=None, ylim=None, xlab=None, **kwargs):
    # arguments:
    # data is a vector of survival times in any order
    # (1-alpha) is the approximate coverage probability for the confidence band

    n = len(data)
    data = sorted(data)
    S = np.zeros(n)
    M = np.zeros(n)
    Fem = np.zeros(n)
    quant = np.zeros(n)
    a = np.zeros(n)
    MU = np.zeros(n)
    ML = np.zeros(n)
    PM = np.zeros(2*n)
    PMU = np.zeros(2*n)
    PML = np.zeros(2*n)

    # calculation of S(x), M(x), and the empirical dsn at the survival times
    S[0] = n
    M[0] = np.mean(data)
    Fem[0] = 1
    for i in range(1, n):
        S[i] = n - i
        M[i] = (sum(data[i:]) / (n - i)) - data[i-1]
        Fem[i] = (n - i) / n

    # Table of critical values for Hall-Wellner confidence band.
    quant = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75])
    a = np.array([2.807, 2.241, 1.96, 1.534, 1.149, 0.871])

    aalpha = a[np.where(quant == alpha)[0][0]]

    Dn = (aalpha * (np.var(data) * len(data) / (len(data) - 1)) ** 0.5) / (n ** 0.5)

    # calculation of the bands ML and MU
    for i in range(n):
        ML[i] = M[i] - (Dn / Fem[i])
        MU[i] = M[i] + (Dn / Fem[i])

    # calculation of PM(x), PMU(x), PML(x) for plotting.
    PM[0] = M[0]
    PM[1] = M[0] - data[0]
    PMU[0] = MU[0]
    PMU[1] = MU[0] - data[0]
    PML[0] = ML[0]
    PML[1] = ML[0] - data[0]

    for i in range(2, n+1):
        PM[2 * i - 2] = M[i - 1]
        PM[2 * i - 1] = M[i - 1] + (data[i - 2] - data[i - 1])
        PMU[2 * i - 2] = MU[i - 1]
        PMU[2 * i - 1] = MU[i - 1] + (data[i - 2] - data[i - 1])
        PML[2 * i - 2] = ML[i - 1]
        PML[2 * i - 1] = ML[i - 1] + (data[i - 2] - data[i - 1])

    if ylim is None:
        ylim = [min(min(PM), min(PMU), min(PML)), max(max(PM), max(PMU), max(PML))]

    if main is None:
        main = "Plot of Mean Residual Life and bounds"

    if xlab is None:
        xlab = "Time"

    x_data = np.linspace(min(data), max(data), len(PM))
    plt.figure()
    plt.plot(x_data, PM, label="Mean Residual Life", **kwargs)
    plt.plot(x_data, PMU, label="Upper Bound", linestyle="--", **kwargs)
    plt.plot(x_data, PML, label="Lower Bound", linestyle="--", linewidth=2, **kwargs)
    plt.xlabel(xlab)
    plt.ylabel("Mean Residual Life")
    plt.title(main)
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()

    return {"PM": PM, "PMU": PMU, "PML": PML}
