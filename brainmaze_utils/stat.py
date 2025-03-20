# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def kl_divergence(mu1, std1, mu2, std2):
    """
    Parametric KL-Divergence between 2 normal 1-D distributions.

    `Normal Distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_


    """
    return 0.5 * ((std1/std2)**2 + ((mu2-mu1)**2 / std2**2) -1 + 2*np.log(std2/std1))


def kl_divergence_mv(mu1, var1, mu2, var2):
    """
    Multidimensional parametric KL-Divergence between 2 normal distributions.

    `KL-Divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions>`_

    `Trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_

    """
    return 0.5 * ((np.trace(np.dot(np.linalg.inv(var2), var1))) + np.dot(np.dot((mu2 - mu1), np.linalg.inv(var2)), (mu2-mu1).T) - mu1.shape[1] + np.log(np.linalg.det(var2)/np.linalg.det(var1)))[0, 0]


def combine_gauss_distributions(mu1, std1, N1, mu2, std2, N2):
    """
    Recalculates a normal 1-D distribution given two subsets of data.
    """

    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    std_combined = np.sqrt(
        (N1*std1**2 + N2*std2**2 + N1*((mu1 - mu_combined)**2) + N2*((mu2 - mu_combined)**2)) / (N1+N2)
    ) #
    # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf - 1.4
    return mu_combined, std_combined


def combine_mvgauss_distributions(mu1, var1, N1, mu2, var2, N2):
    """
    Recalculates a normal n-D distribution given two subsets of data.
    """
    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    var_combined = (N1*(var1) + N2*(var2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2) # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    for k1 in range(var_combined.shape[0]):
        for k2 in range(k1+1, var_combined.shape[0]):
            var_combined[k2, k1] = var_combined[k1, k2]
    return mu_combined, var_combined


def kl_divergence_nonparametric(pk, qk):
    """
    Calculates non-parametric KL-Divergence between two 1-D distributions given by 2 histograms with same bins.
    """
    l_ = pk / qk
    barr = (~np.isinf(l_)) & (~np.isnan(l_))
    return np.nansum(pk[barr] * np.log(l_[barr]))

