import pytest
import pandas as pd
import numpy as np
import math

# Assuming the function is in 'homogeneity_tests.py'
from brainmaze_utils.stats.tests_distribution import variance_homogeneity_f_test

# --- Test Data Fixtures ---

@pytest.fixture
def variance_data():
    """DataFrame for testing variance equality."""
    np.random.seed(456)
    n = 50 # Good sample size for power
    data = {
        'Condition': ['A_eq'] * n + ['B_eq'] * n + ['C_neq'] * n + ['D_insuf'] * 1 + ['E_zero'] * n,
        'Measurement': list(np.random.normal(loc=10, scale=2.0, size=n)) +  # A_eq: std=2
                       list(np.random.normal(loc=15, scale=2.0, size=n)) +  # B_eq: std=2 (Equal to A)
                       list(np.random.normal(loc=20, scale=4.0, size=n)) +  # C_neq: std=4 (Unequal to A/B)
                       [5] + # D_insuf: n=1
                       [30.0] * n # E_zero: variance = 0
    }
    return pd.DataFrame(data)

@pytest.fixture
def variance_data_hue():
    """DataFrame with hue for testing variance equality."""
    np.random.seed(789)
    n = 50
    data_list = []
    # Block X: Control (std=3), Treat (std=3) -> Equal
    data_list.extend([{'Block': 'X', 'Group': 'Control', 'Value': v} for v in np.random.normal(5, 3, n)])
    data_list.extend([{'Block': 'X', 'Group': 'Treat', 'Value': v} for v in np.random.normal(8, 3, n)])
    # Block Y: Control (std=2), Treat (std=5) -> Unequal
    data_list.extend([{'Block': 'Y', 'Group': 'Control', 'Value': v} for v in np.random.normal(10, 2, n)])
    data_list.extend([{'Block': 'Y', 'Group': 'Treat', 'Value': v} for v in np.random.normal(12, 5, n)])
    # Block Z: Control (n<2), Treat (n=OK)
    data_list.extend([{'Block': 'Z', 'Group': 'Control', 'Value': v} for v in [1]])
    data_list.extend([{'Block': 'Z', 'Group': 'Treat', 'Value': v} for v in np.random.normal(15, 1, n)])

    return pd.DataFrame(data_list)


# --- Test Functions ---

def test_f_test_equal_variances(variance_data):
    """Test F-test when variances are expected to be equal."""
    groups_to_compare = ['A_eq', 'B_eq']
    results = variance_homogeneity_f_test(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=groups_to_compare
    )
    assert results is not None
    assert results['group1_id'] == 'A_eq'
    assert results['group2_id'] == 'B_eq'
    assert results['n1'] == 50
    assert results['n2'] == 50
    assert results['p_value'] > 0.05 # Expect non-significant difference


def test_f_test_unequal_variances(variance_data):
    """Test F-test when variances are expected to be unequal."""
    groups_to_compare = ['A_eq', 'C_neq'] # std=2 vs std=4
    results = variance_homogeneity_f_test(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=groups_to_compare
    )
    assert results is not None
    assert results['group1_id'] == 'A_eq'
    assert results['group2_id'] == 'C_neq'
    assert results['n1'] == 50
    assert results['n2'] == 50
    assert results['variance1'] == pytest.approx(4.0, rel=0.3) # Approx checks
    assert results['variance2'] == pytest.approx(16.0, rel=0.3)

    if results['variance1'] > 0 and results['variance2'] > 0:
         expected_f = max(results['variance1'], results['variance2']) / min(results['variance1'], results['variance2'])
         assert results['f_statistic'] == pytest.approx(expected_f)
    else:
         assert math.isnan(results['f_statistic'])

    assert results['p_value'] < 0.05 # Expect significant difference (this assertion should be okay)


def test_f_test_with_hue_equal(variance_data_hue):
    """Test F-test with hue when variances are expected to be equal."""
    results = variance_homogeneity_f_test(
        data=variance_data_hue,
        x='Block',
        y='Value',
        hue='Group',
        order=['X'], # Specify the single x block
        hue_order=['Control', 'Treat'] # Specify the two hue groups
    )
    assert results is not None
    assert results['group1_id'] == ('X', 'Control')
    assert results['group2_id'] == ('X', 'Treat')
    assert results['n1'] == 50
    assert results['n2'] == 50
    assert results['p_value'] > 0.05


def test_f_test_with_hue_unequal(variance_data_hue):
    """Test F-test with hue when variances are expected to be unequal."""
    results = variance_homogeneity_f_test(
        data=variance_data_hue,
        x='Block',
        y='Value',
        hue='Group',
        order=['Y'], # Specify the single x block
        hue_order=['Control', 'Treat'] # Specify the two hue groups (std=2 vs std=5)
    )
    assert results is not None
    assert results['group1_id'] == ('Y', 'Control')
    assert results['group2_id'] == ('Y', 'Treat')
    assert results['n1'] == 50
    assert results['n2'] == 50
    assert results['p_value'] < 0.05


def test_f_test_insufficient_samples_no_hue(variance_data):
    """Test F-test returns None if min_samples not met (no hue)."""
    groups_to_compare = ['A_eq', 'D_insuf'] # n=50 vs n=1
    results = variance_homogeneity_f_test(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=groups_to_compare,
        min_samples=2 # Default
    )
    assert results is None


def test_f_test_insufficient_samples_with_hue(variance_data_hue):
    """Test F-test returns None if min_samples not met (with hue)."""
    results_hue = variance_homogeneity_f_test(
        data=variance_data_hue, # Now correctly uses the fixture result
        x='Block',
        y='Value',
        hue='Group',
        order=['Z'], # Specify the single x block
        hue_order=['Control', 'Treat'] # n=1 vs n=50
    )
    assert results_hue is None


def test_f_test_zero_variance(variance_data):
    """Test F-test handling when one group has zero variance."""
    groups_to_compare = ['A_eq', 'E_zero'] # std=2 vs std=0
    results = variance_homogeneity_f_test(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=groups_to_compare
    )
    assert results is not None
    assert results['variance1'] > 0
    assert results['variance2'] == 0
    assert math.isnan(results['f_statistic']) # F cannot be computed
    assert results['p_value'] == 0.0 # Variances are definitely unequal

    # Test two zero variance groups
    df_zero = pd.DataFrame({
        'Cond': ['G1']*5 + ['G2']*5,
        'Val': [10]*5 + [20]*5
    })
    results_both_zero = variance_homogeneity_f_test(data=df_zero, x='Cond', y='Val', order=['G1', 'G2'])
    assert results_both_zero is not None
    assert results_both_zero['variance1'] == 0
    assert results_both_zero['variance2'] == 0
    assert math.isnan(results_both_zero['f_statistic'])
    assert results_both_zero['p_value'] == 1.0 # Variances are equal (both zero)


def test_f_test_wrong_number_of_groups_specified(variance_data, variance_data_hue):
    """Test ValueError if order/hue_order don't specify exactly two groups."""
    # No hue, order has 1 element
    with pytest.raises(ValueError, match="must be a list containing exactly two category names"):
        variance_homogeneity_f_test(data=variance_data, x='Condition', y='Measurement', order=['A_eq'])

    # No hue, order has 3 elements
    with pytest.raises(ValueError, match="must be a list containing exactly two category names"):
        variance_homogeneity_f_test(data=variance_data, x='Condition', y='Measurement', order=['A_eq', 'B_eq', 'C_neq'])

    # With hue, order has 2 elements
    with pytest.raises(ValueError, match="must be a list containing exactly one category name"):
        variance_homogeneity_f_test(data=variance_data_hue, x='Block', y='Value', hue='Group', order=['X', 'Y'], hue_order=['Control', 'Treat'])

    # With hue, hue_order has 1 element
    with pytest.raises(ValueError, match="must be a list containing exactly two category names"):
         variance_homogeneity_f_test(data=variance_data_hue, x='Block', y='Value', hue='Group', order=['X'], hue_order=['Control'])

    # With hue, hue_order has 3 elements
    with pytest.raises(ValueError, match="must be a list containing exactly two category names"):
         variance_homogeneity_f_test(data=variance_data_hue, x='Block', y='Value', hue='Group', order=['X'], hue_order=['Control', 'Treat', 'Other'])


def test_f_test_nan_handling(variance_data):
    """Test F-test ignores NaNs correctly."""
    variance_data.loc[0, 'Measurement'] = np.nan # Add NaN to A_eq
    variance_data.loc[50, 'Measurement'] = np.nan # Add NaN to B_eq (first element)

    groups_to_compare = ['A_eq', 'B_eq']
    results = variance_homogeneity_f_test(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=groups_to_compare
    )
    assert results is not None
    assert results['n1'] == 49 # Was 50
    assert results['n2'] == 49 # Was 50
    # Variances will change slightly, but should still be approximately equal
    assert results['p_value'] > 0.05 # Expect non-significant difference still




