# tests/test_comparison_tests.py

import pytest
import pandas as pd
import numpy as np
import math

# Assuming function is in normality.py or similar stats module
from brainmaze_utils.stats.tests_unpaired import compare_two_groups_unpaired

# --- Fixtures ---

@pytest.fixture
def comparison_data_normal():
    """Data with two normally distributed groups, different means."""
    np.random.seed(101)
    n = 50
    data = {
        'Category': ['Group1'] * n + ['Group2'] * n,
        'Value': list(np.random.normal(loc=10, scale=2, size=n)) + \
                 list(np.random.normal(loc=12, scale=2, size=n)) # Different means, same var
    }
    return pd.DataFrame(data)

@pytest.fixture
def comparison_data_normal_equal():
    """Data with two normally distributed groups, equal means."""
    np.random.seed(102)
    n = 50
    data = {
        'Category': ['GrpA'] * n + ['GrpB'] * n,
        'Value': list(np.random.normal(loc=20, scale=3, size=n)) + \
                 list(np.random.normal(loc=20, scale=3, size=n)) # Equal means, same var
    }
    return pd.DataFrame(data)

@pytest.fixture
def comparison_data_mixed_dist():
    """Data with one normal, one non-normal (uniform) group."""
    np.random.seed(103)
    n = 50
    data = {
        'Design': ['Norm'] * n + ['Unif'] * n,
        'Score': list(np.random.normal(loc=5, scale=1, size=n)) + \
                 list(np.random.uniform(low=4, high=6, size=n)) # Similar range, diff dist
    }
    return pd.DataFrame(data)

@pytest.fixture
def comparison_data_non_normal():
    """Data with two non-normal (uniform, exponential) groups."""
    np.random.seed(104)
    n = 50
    data = {
        'Type': ['Uniform'] * n + ['Exponential'] * n,
        'Time': list(np.random.uniform(low=10, high=20, size=n)) + \
                list(np.random.exponential(scale=15, size=n)) # Different distributions
    }
    return pd.DataFrame(data)

@pytest.fixture
def comparison_data_hue():
    """Data using hue to define two comparable groups."""
    np.random.seed(105)
    n = 40
    data_list = []
    # Block M, Cond1 (Normal, mean 10)
    data_list.extend([{'Block': 'M', 'Cond': 'C1', 'Val': v} for v in np.random.normal(10, 1, n)])
    # Block M, Cond2 (Normal, mean 12)
    data_list.extend([{'Block': 'M', 'Cond': 'C2', 'Val': v} for v in np.random.normal(12, 1, n)])
    # Block N (ignored in relevant tests)
    data_list.extend([{'Block': 'N', 'Cond': 'C1', 'Val': v} for v in np.random.normal(100, 5, n)])
    return pd.DataFrame(data_list)

@pytest.fixture
def comparison_data_normal_unequal_var():
    """Data with two normally distributed groups, different means, unequal variance."""
    np.random.seed(106)
    n = 50
    data = {
        'Category': ['GroupA'] * n + ['GroupB'] * n,
        'Value': list(np.random.normal(loc=10, scale=2, size=n)) + \
                 list(np.random.normal(loc=12, scale=4, size=n)) # Different means, unequal var (std=2 vs std=4)
    }
    return pd.DataFrame(data)


# --- Test Functions ---

def test_compare_normal_equal_var_diff_means(comparison_data_normal):
    """Test Student's t-test is used for normal, equal variance data with different means."""
    results = compare_two_groups_unpaired(
        data=comparison_data_normal, x='Category', y='Value', order=['Group1', 'Group2']
    )
    assert results is not None
    n1_actual = comparison_data_normal[comparison_data_normal['Category']=='Group1']['Value'].dropna().shape[0]
    n2_actual = comparison_data_normal[comparison_data_normal['Category']=='Group2']['Value'].dropna().shape[0]
    assert results['n1'] == n1_actual
    assert results['n2'] == n2_actual
    assert results['normality_p_value1'] > 0.05
    assert results['normality_p_value2'] > 0.05
    assert results['variance_test_used'] == 'F-test'
    assert results['variance_p_value'] > 0.05 # Expect equal variance
    assert results['test_used'] == 'Student t-test' # Should now use Student's
    assert results['p_value'] < 0.05 # Expect significant difference in means


def test_compare_normal_equal_var_equal_means(comparison_data_normal_equal):
    """Test Student's t-test is used for normal, equal variance data with equal means."""
    results = compare_two_groups_unpaired(
        data=comparison_data_normal_equal, x='Category', y='Value', order=['GrpA', 'GrpB']
    )
    assert results is not None
    assert results['normality_p_value1'] > 0.05
    assert results['normality_p_value2'] > 0.05
    assert results['variance_test_used'] == 'F-test'
    assert results['variance_p_value'] > 0.05 # Expect equal variance
    assert results['test_used'] == 'Student t-test' # Should now use Student's
    assert results['p_value'] > 0.05 # Expect non-significant difference in means

def test_compare_normal_unequal_var(comparison_data_normal_unequal_var):
    """Test Welch's t-test is used for normal, unequal variance data."""
    results = compare_two_groups_unpaired(
        data=comparison_data_normal_unequal_var, x='Category', y='Value', order=['GroupA', 'GroupB']
    )
    assert results is not None
    assert results['normality_p_value1'] > 0.05
    assert results['normality_p_value2'] > 0.05
    assert results['variance_test_used'] == 'F-test'
    assert results['variance_p_value'] < 0.05 # Expect unequal variance
    assert results['test_used'] == 'Welch t-test' # Should use Welch's
    # Means are different (10 vs 12), expect significance
    assert results['p_value'] < 0.05


def test_compare_mixed_distributions_still_mwu(comparison_data_mixed_dist):
    """Test Mann-Whitney U is still used when one group is not normal (variance test is irrelevant here)."""
    results = compare_two_groups_unpaired(
        data=comparison_data_mixed_dist, x='Design', y='Score', order=['Norm', 'Unif']
    )
    assert results is not None
    assert results['normality_p_value1'] > 0.05 # Norm is normal
    assert results['normality_p_value2'] < 0.05 # Unif is not
    # Variance test might run, but shouldn't influence final test choice
    assert results['variance_test_used'] == 'F-test' # Or N/A if samples < 2, but > 3 here
    assert results['test_used'] == 'Mann-Whitney U'
    # Check p-value produced
    assert results['p_value'] is not None

def test_compare_both_non_normal_still_mwu(comparison_data_non_normal):
    """Test Mann-Whitney U is still used when both groups are not normal."""
    results = compare_two_groups_unpaired(
        data=comparison_data_non_normal, x='Type', y='Time', order=['Uniform', 'Exponential']
    )
    assert results is not None
    assert results['normality_p_value1'] < 0.05
    assert results['normality_p_value2'] < 0.05
    assert results['test_used'] == 'Mann-Whitney U'
    assert results['variance_test_used'] == 'F-test'
    assert results['p_value'] < 0.05 # Expect difference

def test_compare_with_hue_checks_variance(comparison_data_hue):
    """Test comparison with hue also checks variance for t-test selection."""
    # Groups ('M', 'C1') and ('M', 'C2') are normal, diff means, equal var (std=1)
    results = compare_two_groups_unpaired(
        data=comparison_data_hue, x='Block', y='Val', hue='Cond',
        order=['M'], hue_order=['C1', 'C2']
    )
    assert results is not None
    assert results['group1_id'] == ('M', 'C1')
    assert results['group2_id'] == ('M', 'C2')
    assert results['normality_p_value1'] > 0.05
    assert results['normality_p_value2'] > 0.05
    assert results['variance_test_used'] == 'F-test'
    assert results['variance_p_value'] > 0.05 # Variances should be equal
    assert results['test_used'] == 'Student t-test' # Should use Student's
    assert results['p_value'] < 0.05 # Means were different (10 vs 12)


def test_compare_mixed_distributions(comparison_data_mixed_dist):
    """Test Mann-Whitney U is used when one group is not normal."""
    results = compare_two_groups_unpaired(
        data=comparison_data_mixed_dist, x='Design', y='Score', order=['Norm', 'Unif']
    )
    assert results is not None
    n_norm_actual = comparison_data_mixed_dist[comparison_data_mixed_dist['Design']=='Norm']['Score'].dropna().shape[0]
    n_unif_actual = comparison_data_mixed_dist[comparison_data_mixed_dist['Design']=='Unif']['Score'].dropna().shape[0]
    assert results['n1'] == n_norm_actual
    assert results['n2'] == n_unif_actual
    assert (results['normality_p_value1'] > 0.05 or results['normality_p_value2'] < 0.05) # One normal, one not
    assert results['test_used'] == 'Mann-Whitney U'
    # P-value assertion depends on whether distributions differ significantly in location
    # In this case, means/medians are similar, so p-value might be > 0.05
    # Let's just check the test used was correct.
    assert results['p_value'] is not None # Check test ran

def test_compare_both_non_normal(comparison_data_non_normal):
    """Test Mann-Whitney U is used when both groups are not normal."""
    results = compare_two_groups_unpaired(
        data=comparison_data_non_normal, x='Type', y='Time', order=['Uniform', 'Exponential']
    )
    assert results is not None
    assert results['normality_p_value1'] < 0.05
    assert results['normality_p_value2'] < 0.05
    assert results['test_used'] == 'Mann-Whitney U'
    assert results['p_value'] < 0.05 # Expect difference due to different distributions

def test_compare_with_hue(comparison_data_hue):
    """Test comparison works correctly when using hue."""
    results = compare_two_groups_unpaired(
        data=comparison_data_hue, x='Block', y='Val', hue='Cond',
        order=['M'], hue_order=['C1', 'C2']
    )
    assert results is not None
    assert results['group1_id'] == ('M', 'C1')
    assert results['group2_id'] == ('M', 'C2')
    assert results['normality_p_value1'] > 0.05
    assert results['normality_p_value2'] > 0.05
    assert results['test_used'] == 'Student t-test'
    assert results['p_value'] < 0.05 # Means were different (10 vs 12)




def test_compare_insufficient_samples(comparison_data_normal):
    """Test returns None if a group has < min_samples_normality."""
    data_small = comparison_data_normal[comparison_data_normal['Category'] == 'Group2'].copy()
    # Add only 2 samples for Group1 (less than min_samples_normality=3)
    data_small = pd.concat([data_small, pd.DataFrame({'Category': ['Group1']*2, 'Value': [1, 1]})], ignore_index=True)

    results = compare_two_groups_unpaired(
        data=data_small, x='Category', y='Value', order=['Group1', 'Group2'], min_samples_normality=3
    )
    # Check the specific failure indicator instead of assert results is None
    assert results is not None
    assert results['test_used'] == 'N/A (Insufficient Samples)'
    assert math.isnan(results['p_value'])
    assert results['n1'] == 2 # Verify the sample size was correctly identified
    assert results['n2'] == 50


def test_compare_not_exactly_two_groups(comparison_data_normal, comparison_data_hue):
    """Test raises ValueError if not exactly two groups are specified."""
    # Only one group in order
    with pytest.raises(ValueError, match="must be a list containing exactly two category names"):
        compare_two_groups_unpaired(data=comparison_data_normal, x='Category', y='Value', order=['Group1'])

    # Three groups in order
    df_3 = pd.concat([comparison_data_normal, pd.DataFrame({'Category':['Group3']*10, 'Value':[1]*10})])
    with pytest.raises(ValueError, match="must be a list containing exactly two category names"):
        compare_two_groups_unpaired(data=df_3, x='Category', y='Value', order=['Group1', 'Group2', 'Group3'])

    match_pattern = "exactly two category names from column 'Cond'"
    with pytest.raises(ValueError, match=match_pattern):
         # Added dummy C3 group data to ensure the groups exist for the check to proceed far enough
         comparison_data_hue_extended = pd.concat([
            comparison_data_hue,
            pd.DataFrame([{'Block': 'M', 'Cond': 'C3', 'Val': 100}])
         ], ignore_index=True)
         compare_two_groups_unpaired(data=comparison_data_hue_extended, x='Block', y='Val', hue='Cond', order=['M'], hue_order=['C1', 'C2', 'C3'])


def test_compare_nan_handling(comparison_data_normal):
    """Test that NaNs are handled correctly before tests."""
    comparison_data_normal.loc[0, 'Value'] = np.nan # Add NaN to Group1
    comparison_data_normal.loc[50, 'Value'] = np.nan # Add NaN to Group2

    results = compare_two_groups_unpaired(
        data=comparison_data_normal, x='Category', y='Value', order=['Group1', 'Group2']
    )
    assert results is not None
    # Dynamically check n after NaN drop
    n1_expected = comparison_data_normal[comparison_data_normal['Category']=='Group1']['Value'].dropna().shape[0]
    n2_expected = comparison_data_normal[comparison_data_normal['Category']=='Group2']['Value'].dropna().shape[0]
    assert results['n1'] == n1_expected # 49
    assert results['n2'] == n2_expected # 49
    assert results['normality_p_value1'] > 0.05
    assert results['normality_p_value2'] > 0.05
    assert results['variance_p_value'] > 0.05 # Variances likely still equal
    assert results['test_used'] == 'Student t-test' # Use Student's
    assert results['p_value'] < 0.05 # Should still detect difference
