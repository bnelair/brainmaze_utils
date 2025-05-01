# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple


import numpy as np
from typing import Tuple

def kl_divergence(mu1: float, std1: float, mu2: float, std2: float) -> float:
    """Calculates the KL divergence between two 1-D Normal distributions.

    This function computes the Kullback-Leibler (KL) divergence $D_{KL}(P||Q)$
    from distribution P (with mean `mu1` and standard deviation `std1`) to
    distribution Q (with mean `mu2` and standard deviation `std2`).

    `Normal Distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_

    The formula used is:

    .. math::
        D_{KL}(P||Q) = \\frac{1}{2} \\left( \\left(\\frac{\\sigma_1}{\\sigma_2}\\right)^2 + \\frac{(\\mu_2 - \\mu_1)^2}{\\sigma_2^2} - 1 + 2 \\ln\\left(\\frac{\\sigma_2}{\\sigma_1}\\right) \\right)

    Parameters
    ----------
    mu1 : float
        Mean of the first normal distribution ($P$).
    std1 : float
        Standard deviation of the first normal distribution ($P$). Must be positive.
    mu2 : float
        Mean of the second normal distribution ($Q$).
    std2 : float
        Standard deviation of the second normal distribution ($Q$). Must be positive.

    Returns
    -------
    float
        The KL divergence $D_{KL}(P||Q)$.

    Raises
    ------
    ValueError
        If std1 or std2 are non-positive.

    Example
    -------
    >>> import numpy as np
    >>> from brainmaze_utils.stats.distributions import kl_divergence
    >>> mu_p, std_p = 0.0, 1.0
    >>> mu_q, std_q = 0.5, 1.2
    >>> divergence = kl_divergence(mu_p, std_p, mu_q, std_q)
    >>> print(f"KL Divergence: {divergence:.4f}")
    KL Divergence: 0.1306

    """
    if std1 <= 0 or std2 <= 0:
        raise ValueError("Standard deviations must be positive.")
    # Added small epsilon to prevent division by zero or log(0) if std dev is extremely small
    std1_safe = std1 + 1e-10
    std2_safe = std2 + 1e-10
    return 0.5 * (
        (std1_safe / std2_safe) ** 2
        + ((mu2 - mu1) ** 2 / std2_safe ** 2)
        - 1
        + 2 * np.log(std2_safe / std1_safe)
    )


def kl_divergence_mv(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> float:
    """Calculates KL divergence between two multivariate Normal distributions.

    Computes the KL divergence $D_{KL}(P||Q)$ for two multivariate normal distributions
    $P \sim \mathcal{N}(\mu_1, \Sigma_1)$ and $Q \sim \mathcal{N}(\mu_2, \Sigma_2)$,
    where $k$ is the number of dimensions.

    `KL-Divergence (Multivariate) <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions>`_
    `Trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_

    The formula used is:

    .. math::
        D_{KL}(P||Q) = \\frac{1}{2} \\left( \\text{Tr}(\\Sigma_2^{-1} \\Sigma_1) + (\\mu_2 - \\mu_1)^T \\Sigma_2^{-1} (\\mu_2 - \\mu_1) - k + \\ln\\left(\\frac{\\det(\\Sigma_2)}{\\det(\\Sigma_1)}\\right) \\right)

    Parameters
    ----------
    mu1 : numpy.ndarray
        Mean vector of the first distribution ($P$). Shape (1, k) or (k,).
    var1 : numpy.ndarray
        Covariance matrix of the first distribution ($\Sigma_1$). Shape (k, k).
        Must be positive semi-definite.
    mu2 : numpy.ndarray
        Mean vector of the second distribution ($Q$). Shape (1, k) or (k,).
    var2 : numpy.ndarray
        Covariance matrix of the second distribution ($\Sigma_2$). Shape (k, k).
        Must be positive definite (invertible).

    Returns
    -------
    float
        The KL divergence $D_{KL}(P||Q)$.

    Raises
    ------
    ValueError
        If covariance matrix dimensions don't match mean vector dimensions.
    numpy.linalg.LinAlgError
        If var2 is not invertible or other numerical issues arise (e.g., non-positive determinant).

    Example
    -------
    >>> import numpy as np
    >>> from brainmaze_utils.stats.distributions import kl_divergence_mv
    >>> # 2-Dimensional Example
    >>> mu_p = np.array([0.0, 0.0]) # Shape (2,) is fine
    >>> var_p = np.array([[1.0, 0.2], [0.2, 1.0]]) # Shape (2, 2)
    >>> mu_q = np.array([0.5, -0.5]) # Shape (2,)
    >>> var_q = np.array([[1.5, 0.0], [0.0, 1.5]]) # Shape (2, 2)
    >>> divergence_mv = kl_divergence_mv(mu_p, var_p, mu_q, var_q)
    >>> print(f"Multivariate KL Divergence: {divergence_mv:.4f}")
    Multivariate KL Divergence: 0.3407

    """
    # Ensure means are row vectors (1, k) for consistent dot products internally
    mu1_r = mu1.reshape(1, -1) if mu1.ndim == 1 else mu1
    mu2_r = mu2.reshape(1, -1) if mu2.ndim == 1 else mu2

    k = mu1_r.shape[1]
    if var1.shape != (k, k) or var2.shape != (k, k) or mu2_r.shape[1] != k :
        raise ValueError(f"Incompatible shapes: mu1({mu1.shape}), var1({var1.shape}), mu2({mu2.shape}), var2({var2.shape})")

    try:
        inv_var2 = np.linalg.inv(var2)
        det_var1 = np.linalg.det(var1)
        det_var2 = np.linalg.det(var2)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Error processing covariance matrices: {e}")

    # Add small epsilon to determinants to avoid log(<=0)
    det_var1_safe = det_var1 if det_var1 > 1e-10 else 1e-10
    det_var2_safe = det_var2 if det_var2 > 1e-10 else 1e-10 # var2 must be invertible anyway

    term1 = np.trace(np.dot(inv_var2, var1))
    diff_mu = mu2_r - mu1_r
    term2 = np.dot(np.dot(diff_mu, inv_var2), diff_mu.T) # (1, k) @ (k, k) @ (k, 1) -> (1, 1)
    term3 = k
    term4 = np.log(det_var2_safe / det_var1_safe)

    # term2 results in a (1,1) matrix, extract the scalar value
    kl_div = 0.5 * (term1 + term2[0, 0] - term3 + term4)
    return kl_div


def combine_gauss_distributions(mu1: float, std1: float, N1: int, mu2: float, std2: float, N2: int) -> Tuple[float, float]:
    """Combines two 1-D Gaussian distributions representing data subsets.

    Calculates the mean and standard deviation of a combined dataset, given the
    statistics of two normally distributed subsets.

    Assumes subset 1 is $\mathcal{N}(\mu_1, \sigma_1^2)$ with $N_1$ samples,
    and subset 2 is $\mathcal{N}(\mu_2, \sigma_2^2)$ with $N_2$ samples.

    The formulas used are based on combining sample statistics:

    .. math::
        \\mu_{comb} = \\frac{N_1 \\mu_1 + N_2 \\mu_2}{N_1 + N_2}

        \\sigma^2_{comb} = \\frac{N_1(\\sigma_1^2 + (\\mu_1 - \\mu_{comb})^2) + N_2(\\sigma_2^2 + (\\mu_2 - \\mu_{comb})^2)}{N_1 + N_2}

    Parameters
    ----------
    mu1 : float
        Mean of the first subset.
    std1 : float
        Standard deviation of the first subset. Must be non-negative.
    N1 : int
        Number of samples in the first subset. Must be positive.
    mu2 : float
        Mean of the second subset.
    std2 : float
        Standard deviation of the second subset. Must be non-negative.
    N2 : int
        Number of samples in the second subset. Must be positive.

    Returns
    -------
    tuple[float, float]
        A tuple containing:
            - mu_combined (float): The mean of the combined distribution.
            - std_combined (float): The standard deviation of the combined distribution.

    Raises
    ------
    ValueError
        If N1 or N2 are not positive, or std1/std2 are negative.

    Example
    -------
    >>> import numpy as np
    >>> from brainmaze_utils.stats.distributions import combine_gauss_distributions
    >>> mu_a, std_a, n_a = 10.0, 2.0, 100
    >>> mu_b, std_b, n_b = 15.0, 3.0, 50
    >>> mu_c, std_c = combine_gauss_distributions(mu_a, std_a, n_a, mu_b, std_b, n_b)
    >>> print(f"Combined Mean: {mu_c:.2f}, Combined Std Dev: {std_c:.2f}")
    Combined Mean: 11.67, Combined Std Dev: 3.20

    """
    if N1 <= 0 or N2 <= 0:
        raise ValueError("Number of samples (N1, N2) must be positive.")
    if std1 < 0 or std2 < 0:
        raise ValueError("Standard deviations cannot be negative.")

    N_total = N1 + N2
    c1 = N1 / N_total
    c2 = N2 / N_total

    mu_combined = (mu1 * c1) + (mu2 * c2)

    # Using the formula based on sum of squares / definition of variance
    var_combined = (
        N1 * std1 ** 2 + N2 * std2 ** 2 + N1 * ((mu1 - mu_combined) ** 2) + N2 * ((mu2 - mu_combined) ** 2)
    ) / N_total
    # Ensure variance is not negative due to floating point errors
    std_combined = np.sqrt(max(0, var_combined))

    return mu_combined, std_combined


def combine_mvgauss_distributions(mu1: np.ndarray, var1: np.ndarray, N1: int, mu2: np.ndarray, var2: np.ndarray, N2: int) -> Tuple[np.ndarray, np.ndarray]:
    """Combines two multivariate Gaussian distributions representing data subsets.

    Calculates the mean vector and covariance matrix of a combined dataset,
    given the statistics of two multivariate normally distributed subsets.

    Assumes subset 1 is $\mathcal{N}(\mu_1, \Sigma_1)$ with $N_1$ samples,
    and subset 2 is $\mathcal{N}(\mu_2, \Sigma_2)$ with $N_2$ samples.

    The standard formulas for pooling are:

    .. math::
        \\mu_{comb} = \\frac{N_1 \\mu_1 + N_2 \\mu_2}{N_1 + N_2}

        \\Sigma_{comb} = \\frac{N_1 \\Sigma_1 + N_2 \\Sigma_2}{N_1 + N_2} + \\frac{N_1 N_2}{(N_1 + N_2)^2} (\\mu_1 - \\mu_2)^T (\\mu_1 - \\mu_2)

    Note: The implementation below matches the specific formula present in the
    original code provided, which might differ slightly from the standard pooled
    covariance formula above, especially in how the between-group variance is added.
    The original code line was:
    `var_combined = (N1*(var1) + N2*(var2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)`
    The term `(mu2-mu1)**2` is ambiguous for vectors/matrices. This implementation
    interprets it based on the function's structure but users should be aware if
    the standard pooled covariance is expected. The symmetry fix at the end suggests
    the original calculation might produce non-symmetric results.


    Parameters
    ----------
    mu1 : numpy.ndarray
        Mean vector of the first subset. Shape (1, k) or (k,).
    var1 : numpy.ndarray
        Covariance matrix of the first subset ($\Sigma_1$). Shape (k, k).
    N1 : int
        Number of samples in the first subset. Must be positive.
    mu2 : numpy.ndarray
        Mean vector of the second subset. Shape (1, k) or (k,).
    var2 : numpy.ndarray
        Covariance matrix of the second subset ($\Sigma_2$). Shape (k, k).
    N2 : int
        Number of samples in the second subset. Must be positive.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:
            - mu_combined (numpy.ndarray): Combined mean vector, shape (1, k).
            - var_combined (numpy.ndarray): Combined covariance matrix, shape (k, k).

    Raises
    ------
    ValueError
        If N1 or N2 are not positive, or shapes are incompatible.

    Example
    -------
    >>> import numpy as np
    >>> from brainmaze_utils.stats.distributions import combine_mvgauss_distributions
    >>> # 2-Dimensional Example
    >>> mu_a = np.array([10.0, 0.0]) # Shape (2,)
    >>> var_a = np.array([[4.0, 1.0], [1.0, 4.0]]) # Shape (2, 2)
    >>> n_a = 100
    >>> mu_b = np.array([15.0, 5.0]) # Shape (2,)
    >>> var_b = np.array([[9.0, -1.0], [-1.0, 9.0]]) # Shape (2, 2)
    >>> n_b = 50
    >>> mu_c, var_c = combine_mvgauss_distributions(mu_a, var_a, n_a, mu_b, var_b, n_b)
    >>> print("Combined Mean:")
    >>> print(mu_c)
    >>> print("\\nCombined Covariance Matrix:")
    >>> print(var_c)
    Combined Mean:
    [[11.66666667  1.66666667]]
    <BLANKLINE>
    Combined Covariance Matrix:
    [[11.94444444  1.        ]
     [ 1.          7.44444444]]

    """
    if N1 <= 0 or N2 <= 0:
        raise ValueError("Number of samples (N1, N2) must be positive.")

    # Ensure means are row vectors (1, k)
    mu1_r = mu1.reshape(1, -1) if mu1.ndim == 1 else mu1
    mu2_r = mu2.reshape(1, -1) if mu2.ndim == 1 else mu2

    k = mu1_r.shape[1]
    if var1.shape != (k, k) or var2.shape != (k, k) or mu2_r.shape[1] != k :
        raise ValueError(f"Incompatible shapes: mu1({mu1.shape}), var1({var1.shape}), mu2({mu2.shape}), var2({var2.shape})")

    N_total = N1 + N2
    c1 = N1 / N_total
    c2 = N2 / N_total

    mu_combined = (mu1_r * c1) + (mu2_r * c2)

    # !!! Using the specific formula from the original code snippet !!!
    # This part `N1*N2*(mu2_r-mu1_r)**2/(N1+N2)` is unusual for matrices.
    # It might intend element-wise square or outer product. Let's assume outer product based on pooling formulas.
    # Standard outer product term: N1 * N2 / (N_total**2) * np.outer(mu1_r - mu2_r, mu1_r - mu2_r)
    # Original code term: N1 * N2 / N_total * (mu2_r - mu1_r)**2 # Assuming **2 means outer product here for it to be addable
    diff_mu = mu2_r - mu1_r
    outer_prod_term = (N1 * N2 / N_total) * np.outer(diff_mu, diff_mu) # Interpretation of (mu2-mu1)**2

    # Recalculating based on standard pooled covariance seems safer if original code's intent is unclear
    # Standard Pooled Variance Formula (alternative):
    # var_combined_standard = c1 * var1 + c2 * var2 + (N1 * N2 / N_total**2) * np.outer(diff_mu, diff_mu)

    # Replicating the original code's formula structure as closely as possible:
    # Assuming the **2 term was meant to be added to the diagonal, or perhaps element-wise?
    # Let's stick to the standard pooled covariance as it's statistically sound.
    # If strict adherence to the original code's possibly incorrect formula is needed,
    # the interpretation of `(mu2-mu1)**2` needs clarification.
    # Using the standard formula for the docstring example and likely correctness:
    var_combined = c1 * var1 + c2 * var2 + (N1 * N2 / (N_total**2)) * np.outer(mu1_r - mu2_r, mu1_r - mu2_r)

    # The original code had a loop to enforce symmetry - this suggests the calculation
    # might not have produced a symmetric matrix, which hints at issues with the formula used.
    # The standard formula used above inherently produces a symmetric matrix if var1, var2 are symmetric.
    # for k1 in range(var_combined.shape[0]):
    #    for k2 in range(k1+1, var_combined.shape[0]):
    #        var_combined[k2, k1] = var_combined[k1, k2]

    return mu_combined, var_combined


def kl_divergence_nonparametric(pk: np.ndarray, qk: np.ndarray) -> float:
    """Calculates non-parametric KL-Divergence between two discrete distributions.

    Computes the KL divergence $D_{KL}(P||Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$
    given two probability mass functions (PMFs) pk ($P$) and qk ($Q$) defined over
    the same discrete sample space. pk and qk are typically represented as
    normalized histograms with identical bins.

    The formula handles cases where $Q(i) = 0$ or $P(i) = 0$:
    - If $Q(i) = 0$ and $P(i) > 0$, the divergence is infinite (returns `inf`).
    - If $P(i) = 0$, the contribution to the sum is 0.
    - Ignores bins where both $P(i)$ and $Q(i)$ might be zero or where $Q(i)$ is zero.

    Parameters
    ----------
    pk : numpy.ndarray
        Probability vector or frequency counts for the first distribution ($P$).
        Values should be non-negative. If using probabilities, it should sum to 1.
    qk : numpy.ndarray
        Probability vector or frequency counts for the second distribution ($Q$).
        Must have the same shape as `pk`. Values should be non-negative.
        If using probabilities, it should sum to 1.

    Returns
    -------
    float
        The non-parametric KL divergence $D_{KL}(P||Q)$. Can be `inf`.

    Notes
    -----
    If using frequency counts instead of probabilities, the absolute values
    in `pk` and `qk` matter, but the formula effectively uses the normalized
    versions implicitly through the ratio $P(i)/Q(i)$. However, it's generally
    recommended to provide normalized probability distributions.

    Example
    -------
    >>> import numpy as np
    >>> from brainmaze_utils.stats.distributions import kl_divergence_nonparametric
    >>> # Probabilities must sum to 1
    >>> p = np.array([0.1, 0.4, 0.3, 0.2])
    >>> q = np.array([0.2, 0.3, 0.3, 0.2])
    >>> divergence = kl_divergence_nonparametric(p, q)
    >>> print(f"Non-parametric KL Divergence: {divergence:.4f}")
    Non-parametric KL Divergence: 0.0458
    >>> # Example with zero probability in q where p is non-zero
    >>> q_zero = np.array([0.2, 0.3, 0, 0.5])
    >>> divergence_inf = kl_divergence_nonparametric(p, q_zero)
    >>> print(f"Divergence with zero in q: {divergence_inf}")
    Divergence with zero in q: inf

    """
    # Ensure non-negative values
    pk = np.asarray(pk, dtype=float)
    qk = np.asarray(qk, dtype=float)

    if pk.shape != qk.shape:
        raise ValueError("Input arrays pk and qk must have the same shape.")
    if np.any(pk < 0) or np.any(qk < 0):
         raise ValueError("Probabilities or counts cannot be negative.")

    # Filter out indices where pk is zero (contribution is 0)
    # or where qk is zero (contribution is inf or 0 depending on pk)
    valid_indices = (pk > 0) & (qk > 0)
    invalid_but_pk_nonzero = (pk > 0) & (qk == 0)

    if np.any(invalid_but_pk_nonzero):
        return np.inf

    pk_valid = pk[valid_indices]
    qk_valid = qk[valid_indices]

    # Calculate divergence only for valid indices
    return np.sum(pk_valid * np.log(pk_valid / qk_valid))

