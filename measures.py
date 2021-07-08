"""
# Zachary Kimo Stine,
# updated 2021-01-06
#
# Functions for calculating information theoretic quantities. Ignore the checksum arguments, which 
# previously served a use, but no longer do and I need to trackdown each time their used throughout the 
# other scripts and remove them there before removing them from the functions here.
"""
import numpy as np
import math


def entropy(x, checksum=True):
    """
	Shannon entropy of x.
	"""
    if np.around(np.sum(x), decimals=6) != 1.0:
        x = x / x.sum()

    # Calculate Shannon entropy, assuming 0log(0) == 0 and so can be skipped.
    h = 0.0
    for i in np.nditer(x):
        if i > 0.0:
            h += i * math.log(i, 2)
    return -h


def js_divergence(x, y, checksum=True):
    """
	Calculates JSD(x || y).
	"""

    assert len(x) == len(y), 'js_divergence(): x and y do not have the same number of elements.'

    m = np.mean([x, y], axis=0, dtype=np.float64)

    h_m = entropy(m, checksum)
    h_x = entropy(x, checksum)
    h_y = entropy(y, checksum)

    return h_m - ((h_x + h_y) / 2.0)


def kl_divergence(x, y):
    """
	Calculates KLD(x || y)
	"""
    assert len(x) == len(y), 'kl_divergence(): x and y do not have the same number of elements.'

    kld = 0.0
    for i in range(len(x)):
        if x[i] != 0.0:
            kld += x[i] * math.log(x[i] / y[i], 2)
    return kld


def per_word_js_divergence(x, y):
    """
	Calculates each element's contribution to JSD(x || y).
	"""
    assert len(x) == len(y)

    m = np.mean([x, y], axis=0, dtype=np.float64)

    jsd_list = []

    for i in range(len(x)):
        if m[i] != 0.0:
            hm = -1.0 * m[i] * math.log(m[i], 2)
        else:
            hm = 0.0

        if x[i] != 0.0:
            hx = -1.0 * x[i] * math.log(x[i], 2)
        else:
            hx = 0.0

        if y[i] != 0.0:
            hy = -1.0 * y[i] * math.log(y[i], 2)
        else:
            hy = 0.0

        jsd_list.append(hm - ((hx + hy) / 2.0))

    jsd_array = np.array(jsd_list)

    return jsd_array


def per_word_kl_divergence(x, y):
    """
    Calculates each element's contribution to the total KLD(x ||y).
    """
    assert len(x) == len(y)
    kl_list = []

    for i in range(len(x)):
        if x[i] == 0.0:
            kl_list.append(0.0)

        elif x[i] > 0.0 and y[i] == 0.0:
            kl_list.append(float('nan'))

        else:
            kl_list.append(x[i] * math.log(x[i] / y[i], 2))

    return np.array(kl_list)
