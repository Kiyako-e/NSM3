# -*- coding: utf-8 -*-
"""Untitled35.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nlsNotAsfgjvNKuVG7l6L5uaYyHjbek8
"""

import math

def CorrUpperBound(n):
    mu_f_upper = (math.sqrt(2) + 6) / 24
    lambda_f_upper = 7 / 24

    numerator = ((24 * lambda_f_upper - 6) * n**2 +
                 (48 * mu_f_upper - 72 * lambda_f_upper + 7) * n +
                 (48 * lambda_f_upper - 48 * mu_f_upper + 1))
    denominator = (n + 1) * (2 * n + 1)

    return numerator / denominator