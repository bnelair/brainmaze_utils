import pytest
import pandas as pd
import numpy as np

from brainmaze_utils.stats.tests_distribution import assess_homogeneity_brown_forsythe


# --- Fixtures specific to Brown-Forsythe (if needed) ---
@pytest.fixture
def variance_data_3eq(variance_data): # Requires variance_data from conftest.py
    """Adds a third group 'F_eq' with same variance as A_eq, B_eq."""
    np.random.seed(111)
    # Dynamically find n from the fixture data
    n = variance_data[variance_data['Condition']=='A_eq'].shape[0]
    if n == 0: n = 50 # Fallback if A_eq somehow missing
    df_add = pd.DataFrame({
         'Condition': ['F_eq'] * n,
         'Measurement': np.random.normal(loc=25, scale=2.0, size=n) # std=2
    })
    return pd.concat([variance_data, df_add], ignore_index=True)

@pytest.fixture
def variance_data_hue_3eq(variance_data_hue): # Requires variance_data_hue from conftest.py
    """Adds a 'Placebo' group to Block X with same variance."""
    np.random.seed(222)
    # Dynamically find n
    n = variance_data_hue[(variance_data_hue['Block']=='X') & (variance_data_hue['Group']=='Control')].shape[0]
    if n == 0: n = 50 # Fallback
    df_add = pd.DataFrame([
        {'Block': 'X', 'Group': 'Placebo', 'Value': v}
        for v in np.random.normal(6, 3, n) # std=3, same as X/Control and X/Treat
    ])
    return pd.concat([variance_data_hue, df_add], ignore_index=True)


# --- Test Functions for Brown-Forsythe ---

def test_bf_equal_variances_multi_group(variance_data_3eq):
    """Test Brown-Forsythe when >2 groups have equal variances."""
    groups_to_test = ['A_eq', 'B_eq', 'F_eq']
    results = assess_homogeneity_brown_forsythe(
        data=variance_data_3eq,
        x='Condition',
        y='Measurement',
        order=groups_to_test
    )
    assert results is not None
    assert results['n_groups_tested'] == 3
    assert set(results['groups_tested']) == set(groups_to_test) # Order agnostic check
    assert results['p_value'] > 0.05 # Expect non-significant difference

def test_bf_unequal_variances_multi_group(variance_data_3eq):
    """Test Brown-Forsythe when >2 groups have unequal variances."""
    groups_to_test = ['A_eq', 'B_eq', 'C_neq'] # Two with std=2, one with std=4
    results = assess_homogeneity_brown_forsythe(
        data=variance_data_3eq,
        x='Condition',
        y='Measurement',
        order=groups_to_test
    )
    assert results is not None
    assert results['n_groups_tested'] == 3
    assert set(results['groups_tested']) == set(groups_to_test)
    assert results['p_value'] < 0.05 # Expect significant difference

def test_bf_with_hue_equal_multi_group(variance_data_hue_3eq):
    """Test Brown-Forsythe with hue when >2 groups have equal variances."""
    results = assess_homogeneity_brown_forsythe(
        data=variance_data_hue_3eq,
        x='Block',
        y='Value',
        hue='Group',
        order=['X'], # Test within Block X
        # hue_order defaults to testing all available hues for X
    )
    assert results is not None
    # Dynamically find expected groups for Block X
    expected_groups = [
        ('X', g) for g in sorted(variance_data_hue_3eq[variance_data_hue_3eq['Block']=='X']['Group'].unique())
    ]
    assert results['n_groups_tested'] == len(expected_groups)
    assert set(results['groups_tested']) == set(expected_groups)
    assert results['p_value'] > 0.05

def test_bf_with_hue_unequal_multi_group(variance_data_hue):
    """Test Brown-Forsythe with hue when >2 groups have unequal variances."""
    # Add a third group to Block Y to make it a multi-group comparison
    np.random.seed(333)
    n = variance_data_hue[(variance_data_hue['Block']=='Y') & (variance_data_hue['Group']=='Control')].shape[0]
    if n == 0: n = 50 # Fallback
    df_add = pd.DataFrame([
        {'Block': 'Y', 'Group': 'Placebo', 'Value': v}
        for v in np.random.normal(11, 2, n) # std=2 (same as Control, different from Treat)
    ])
    test_data = pd.concat([variance_data_hue, df_add], ignore_index=True)

    results = assess_homogeneity_brown_forsythe(
        data=test_data,
        x='Block',
        y='Value',
        hue='Group',
        order=['Y'], # Test within Block Y
        # hue_order defaults to testing all available groups: Control(2), Treat(5), Placebo(2)
    )
    assert results is not None
    expected_groups = [
        ('Y', g) for g in sorted(test_data[test_data['Block']=='Y']['Group'].unique())
    ]
    assert results['n_groups_tested'] == len(expected_groups) # Should be 3
    assert set(results['groups_tested']) == set(expected_groups)
    assert results['p_value'] < 0.05 # Variances are unequal (2, 5, 2)

def test_bf_fewer_than_2_valid_groups(variance_data):
    """Test Brown-Forsythe returns None if < 2 groups meet min_samples."""
    results = assess_homogeneity_brown_forsythe(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=['A_eq', 'D_insuf'], # Only A_eq meets min_samples=3
        min_samples=3
    )
    assert results is None

    results_only_one = assess_homogeneity_brown_forsythe(
        data=variance_data,
        x='Condition',
        y='Measurement',
        order=['A_eq'], # Only one group specified
        min_samples=3
    )
    assert results_only_one is None

def test_bf_ignores_small_groups(variance_data_3eq):
    """Test that groups below min_samples are excluded from the test."""
    groups_to_include = ['A_eq', 'B_eq', 'D_insuf', 'F_eq'] # D is n=1
    results = assess_homogeneity_brown_forsythe(
        data=variance_data_3eq,
        x='Condition',
        y='Measurement',
        order=groups_to_include,
        min_samples=3 # D_insuf should be excluded
    )
    assert results is not None
    assert results['n_groups_tested'] == 3 # Only A, B, F included
    expected_groups = ['A_eq', 'B_eq', 'F_eq']
    assert set(results['groups_tested']) == set(expected_groups)
    assert results['p_value'] > 0.05 # A, B, F have equal variance

def test_bf_nan_handling(variance_data_3eq):
    """Test Brown-Forsythe ignores NaNs correctly."""
    variance_data_3eq.loc[0, 'Measurement'] = np.nan   # Nan in A_eq
    variance_data_3eq.loc[50, 'Measurement'] = np.nan  # Nan in B_eq
    variance_data_3eq.loc[100, 'Measurement'] = np.nan # Nan in C_neq

    # Test equal variance case with NaNs
    groups_eq = ['A_eq', 'B_eq', 'F_eq']
    results_eq = assess_homogeneity_brown_forsythe(
        data=variance_data_3eq, x='Condition', y='Measurement', order=groups_eq
    )
    assert results_eq is not None
    assert results_eq['n_groups_tested'] == 3
    # Dynamically check groups tested (ensure correct ones despite NaNs)
    assert set(results_eq['groups_tested']) == set(groups_eq)
    assert results_eq['p_value'] > 0.05 # Still expect equal variance

    # Test unequal variance case with NaNs
    groups_neq = ['A_eq', 'C_neq', 'F_eq'] # A(2), C(4), F(2) -> unequal overall
    results_neq = assess_homogeneity_brown_forsythe(
        data=variance_data_3eq, x='Condition', y='Measurement', order=groups_neq
    )
    assert results_neq is not None
    assert results_neq['n_groups_tested'] == 3
    assert set(results_neq['groups_tested']) == set(groups_neq)
    assert results_neq['p_value'] < 0.05 # Still expect unequal variance
