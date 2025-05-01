# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import pytest

from brainmaze_utils.stats.distributions import (
    kl_divergence,
    kl_divergence_mv,
    combine_gauss_distributions,
    combine_mvgauss_distributions,
    kl_divergence_nonparametric
)

# Tolerance for floating point comparisons
RTOL = 1e-5
ATOL = 1e-8

# ==================================
# Tests for kl_divergence (1D Normal)
# ==================================

def test_kl_divergence_identical():
    """
    Test KL divergence between identical 1D normal distributions.
    Rationale: KL(P||P) should always be 0.
    """
    mu, std = 5.0, 2.0
    assert kl_divergence(mu, std, mu, std) == pytest.approx(0.0, abs=ATOL)

def test_kl_divergence_known_values():
    """
    Test KL divergence for a known case against a pre-calculated value.
    Rationale: Verifies the implementation of the KL formula.
    Uses the example from the docstring.
    """
    mu1, std1 = 0.0, 1.0
    mu2, std2 = 0.5, 1.2
    # Reference value calculated from formula / docstring example
    expected_kl = 0.11634933455217225
    assert kl_divergence(mu1, std1, mu2, std2) == pytest.approx(expected_kl, rel=RTOL, abs=ATOL)

def test_kl_divergence_invalid_std_dev():
    """
    Test KL divergence raises ValueError for non-positive standard deviation.
    Rationale: Standard deviation must be positive for the formula to be valid.
    """
    mu1, std1 = 0.0, 1.0
    mu2, std2 = 0.5, 1.2
    with pytest.raises(ValueError, match="Standard deviations must be positive"):
        kl_divergence(mu1, 0.0, mu2, std2) # std1 = 0
    with pytest.raises(ValueError, match="Standard deviations must be positive"):
        kl_divergence(mu1, std1, mu2, -1.0) # std2 < 0

# =======================================
# Tests for kl_divergence_mv (Multivariate Normal)
# =======================================

def test_kl_divergence_mv_identical():
    """
    Test KL divergence between identical multivariate normal distributions.
    Rationale: KL(P||P) should always be 0 for multivariate case as well.
    """
    mu = np.array([1.0, -2.0])
    var = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert kl_divergence_mv(mu, var, mu, var) == pytest.approx(0.0, abs=ATOL)

def test_kl_divergence_mv_known_values():
    """
    Test multivariate KL divergence for a known case (docstring example).
    Rationale: Verifies the implementation of the multivariate KL formula.
    """
    mu_p = np.array([0.0, 0.0])
    var_p = np.array([[1.0, 0.2], [0.2, 1.0]])
    mu_q = np.array([0.5, -0.5])
    var_q = np.array([[1.5, 0.0], [0.0, 1.5]])
    # Reference value calculated from formula / docstring example
    expected_kl_mv = 0.2592094387016252
    assert kl_divergence_mv(mu_p, var_p, mu_q, var_q) == pytest.approx(expected_kl_mv, rel=RTOL, abs=ATOL)

def test_kl_divergence_mv_singular_var2():
    """
    Test multivariate KL divergence raises LinAlgError if var2 is singular (non-invertible).
    Rationale: The formula requires the inverse and determinant of var2.
    """
    mu1 = np.array([0.0])
    var1 = np.array([[1.0]])
    mu2 = np.array([0.0])
    var2 = np.array([[0.0]]) # Singular matrix
    with pytest.raises(np.linalg.LinAlgError):
         # The error might occur during inv() or det() depending on implementation checks
         kl_divergence_mv(mu1, var1, mu2, var2)

def test_kl_divergence_mv_shape_mismatch():
    """
    Test multivariate KL divergence raises ValueError for shape mismatches.
    Rationale: Mean vectors and covariance matrices must have compatible dimensions.
    """
    mu1_2d = np.array([0.0, 0.0])
    var1_1d = np.array([[1.0]])
    var1_2d = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2_2d = np.array([1.0, 1.0])
    var2_2d = np.array([[2.0, 0.0], [0.0, 2.0]])

    with pytest.raises(ValueError, match="Incompatible shapes"):
        # mu1 (2D) incompatible with var1 (1D)
        kl_divergence_mv(mu1_2d, var1_1d, mu2_2d, var2_2d)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        # var1 (2D) incompatible with var2 (different shape, e.g. 1D)
        kl_divergence_mv(mu1_2d, var1_2d, mu2_2d, var1_1d)


# ========================================
# Tests for combine_gauss_distributions (1D)
# ========================================

def test_combine_gauss_identical():
    """
    Test combining two identical 1D Gaussian distributions.
    Rationale: Combining two identical groups should result in the same distribution parameters.
    """
    mu, std, N = 5.0, 1.5, 100
    combined_mu, combined_std = combine_gauss_distributions(mu, std, N, mu, std, N)
    assert combined_mu == pytest.approx(mu, rel=RTOL, abs=ATOL)
    assert combined_std == pytest.approx(std, rel=RTOL, abs=ATOL)

def test_combine_gauss_known_values():
    """
    Test combining two different 1D Gaussians against a known result (docstring example).
    Rationale: Verifies the implementation of the combination formulas.
    """
    mu_a, std_a, n_a = 10.0, 2.0, 100
    mu_b, std_b, n_b = 15.0, 3.0, 50
    expected_mu = 11.66666667
    expected_std = 3.34995854037363 # sqrt(((100*(4+ (10-11.666)**2)) + 50*(9 + (15-11.666)**2)) / 150)

    combined_mu, combined_std = combine_gauss_distributions(mu_a, std_a, n_a, mu_b, std_b, n_b)
    assert combined_mu == pytest.approx(expected_mu, rel=RTOL, abs=ATOL)
    assert combined_std == pytest.approx(expected_std, rel=RTOL, abs=ATOL)

def test_combine_gauss_zero_std():
    """
    Test combining 1D Gaussians where standard deviations are zero.
    Rationale: Checks if the combination handles zero variance correctly.
    The combined std dev should reflect only the variance *between* the means.
    """
    mu1, std1, N1 = 10.0, 0.0, 10
    mu2, std2, N2 = 12.0, 0.0, 10
    # Expected mu = (10*10 + 12*10)/20 = 11
    # Expected var = (10*(0 + (10-11)**2) + 10*(0 + (12-11)**2)) / 20 = (10*1 + 10*1) / 20 = 1
    # Expected std = sqrt(1) = 1
    expected_mu, expected_std = 11.0, 1.0
    combined_mu, combined_std = combine_gauss_distributions(mu1, std1, N1, mu2, std2, N2)
    assert combined_mu == pytest.approx(expected_mu, rel=RTOL, abs=ATOL)
    assert combined_std == pytest.approx(expected_std, rel=RTOL, abs=ATOL)

def test_combine_gauss_invalid_input():
    """
    Test combine_gauss_distributions raises ValueError for invalid N or std dev.
    Rationale: Ensures input validation for sample sizes and std dev works.
    """
    mu1, std1, N1 = 10.0, 1.0, 10
    mu2, std2, N2 = 12.0, 1.0, 10

    with pytest.raises(ValueError, match="Number of samples .* must be positive"):
        combine_gauss_distributions(mu1, std1, 0, mu2, std2, N2) # N1=0
    with pytest.raises(ValueError, match="Number of samples .* must be positive"):
        combine_gauss_distributions(mu1, std1, N1, mu2, std2, -5) # N2 < 0
    with pytest.raises(ValueError, match="Standard deviations cannot be negative"):
        combine_gauss_distributions(mu1, -1.0, N1, mu2, std2, N2) # std1 < 0

# =============================================
# Tests for combine_mvgauss_distributions (Multivariate)
# =============================================

def test_combine_mvgauss_identical():
    """
    Test combining two identical multivariate Gaussian distributions.
    Rationale: Combining identical groups should yield the same parameters.
    """
    mu = np.array([1.0, -1.0])
    var = np.array([[2.0, 0.1], [0.1, 1.0]])
    N = 50
    combined_mu, combined_var = combine_mvgauss_distributions(mu, var, N, mu, var, N)
    assert np.allclose(combined_mu.flatten(), mu, rtol=RTOL, atol=ATOL)
    assert np.allclose(combined_var, var, rtol=RTOL, atol=ATOL)

def test_combine_mvgauss_known_values():
    """
    Test combining two different multivariate Gaussians (docstring example).
    Rationale: Verifies the implementation using the standard pooled statistics formulas.
    """
    mu_a = np.array([10.0, 0.0])
    var_a = np.array([[4.0, 1.0], [1.0, 4.0]])
    n_a = 100
    mu_b = np.array([15.0, 5.0])
    var_b = np.array([[9.0, -1.0], [-1.0, 9.0]])
    n_b = 50

    expected_mu = np.array([11.66666667, 1.66666667])
    # CORRECTED expected covariance matrix
    expected_var = np.array([[11.22222222, 5.88888889],
                             [5.88888889, 11.22222222]])

    combined_mu, combined_var = combine_mvgauss_distributions(mu_a, var_a, n_a, mu_b, var_b, n_b)
    assert np.allclose(combined_mu.flatten(), expected_mu, rtol=RTOL, atol=ATOL)
    assert np.allclose(combined_var, expected_var, rtol=RTOL, atol=ATOL)

def test_combine_mvgauss_invalid_input():
    """
    Test combine_mvgauss_distributions raises ValueError for invalid N or shape mismatch.
    Rationale: Ensure input validation for sample sizes and dimensions works.
    """
    mu1_2d = np.array([0.0, 0.0])
    var1_2d = np.array([[1.0, 0.0], [0.0, 1.0]])
    N1 = 10
    mu2_2d = np.array([1.0, 1.0])
    var2_2d = np.array([[2.0, 0.0], [0.0, 2.0]])
    N2 = 10
    var2_1d = np.array([[1.0]]) # Mismatched shape

    with pytest.raises(ValueError, match="Number of samples .* must be positive"):
        combine_mvgauss_distributions(mu1_2d, var1_2d, 0, mu2_2d, var2_2d, N2) # N1=0
    with pytest.raises(ValueError, match="Incompatible shapes"):
        combine_mvgauss_distributions(mu1_2d, var1_2d, N1, mu2_2d, var2_1d, N2) # var2 shape mismatch

# ============================================
# Tests for kl_divergence_nonparametric
# ============================================

def test_kl_nonparametric_identical():
    """
    Test non-parametric KL divergence for identical distributions.
    Rationale: KL(P||P) should be 0.
    """
    p = np.array([0.1, 0.2, 0.7])
    q = np.array([0.1, 0.2, 0.7])
    assert kl_divergence_nonparametric(p, q) == pytest.approx(0.0, abs=ATOL)

def test_kl_nonparametric_known_value():
    """
    Test non-parametric KL divergence for a known case (docstring example).
    Rationale: Verifies the summation and log calculation.
    """
    p = np.array([0.1, 0.4, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.3, 0.2])
    # 0.1*log(0.1/0.2) + 0.4*log(0.4/0.3) + 0.3*log(0.3/0.3) + 0.2*log(0.2/0.2)
    # = 0.1*log(0.5) + 0.4*log(4/3) + 0 + 0
    expected_kl = 0.04575811092471789
    assert kl_divergence_nonparametric(p, q) == pytest.approx(expected_kl, rel=RTOL, abs=ATOL)

def test_kl_nonparametric_q_zero():
    """
    Test non-parametric KL divergence returns infinity if qk[i]=0 where pk[i]>0.
    Rationale: Division by zero in the log term leads to infinite divergence.
    """
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0]) # q[1]=0 where p[1]>0
    assert kl_divergence_nonparametric(p, q) == np.inf

def test_kl_nonparametric_p_zero():
    """
    Test non-parametric KL divergence handles pk[i]=0 correctly.
    Rationale: Terms where pk[i]=0 should contribute zero to the sum.
    """
    p = np.array([0.0, 1.0])
    q = np.array([0.5, 0.5])
    # 0 * log(0/0.5) + 1 * log(1.0/0.5) = 0 + log(2)
    expected_kl = np.log(2)
    assert kl_divergence_nonparametric(p, q) == pytest.approx(expected_kl, rel=RTOL, abs=ATOL)

def test_kl_nonparametric_counts():
    """
    Test non-parametric KL divergence with unnormalized counts.
    Rationale: Verify the function calculates sum(pk_i * log(pk_i / qk_i)) using raw counts.
    Note: This value is generally NOT the same as KL divergence of the normalized distributions.
    """
    pk_counts = np.array([1, 4, 3, 2]) # Sum = 10
    qk_counts = np.array([2, 3, 3, 2]) # Sum = 10
    # 1*log(1/2) + 4*log(4/3) + 3*log(3/3) + 2*log(2/2)
    # = log(0.5) + 4*log(4/3) + 0 + 0
    expected_val = np.log(0.5) + 4 * np.log(4/3) # Approx 0.457656
    assert kl_divergence_nonparametric(pk_counts, qk_counts) == pytest.approx(expected_val, rel=RTOL, abs=ATOL)

def test_kl_nonparametric_invalid_input():
    """
    Test non-parametric KL divergence raises ValueError for mismatched shapes or negative values.
    Rationale: Ensures input validation.
    """
    p = np.array([0.1, 0.9])
    q_long = np.array([0.1, 0.8, 0.1])
    q_neg = np.array([-0.1, 1.1])

    with pytest.raises(ValueError, match="must have the same shape"):
        kl_divergence_nonparametric(p, q_long)
    with pytest.raises(ValueError, match="cannot be negative"):
        kl_divergence_nonparametric(p, q_neg)
    with pytest.raises(ValueError, match="cannot be negative"):
        kl_divergence_nonparametric(q_neg, p)