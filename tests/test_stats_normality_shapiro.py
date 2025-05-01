import pytest
import pandas as pd
import numpy as np
import math # For math.isnan

# Assuming the function is in 'normality_tests.py'
from brainmaze_utils.stats.tests_distribution import normality_shapiro_wilk_test

# --- Test Data Fixtures ---

@pytest.fixture
def sample_data_no_hue():
    """DataFrame with one grouping variable (x), larger N for A & B"""
    np.random.seed(42) # Keep seed for reproducibility
    n_samples_good = 100 # Increased sample size
    data = {
        'Group': ['A'] * n_samples_good + ['B'] * n_samples_good + ['C'] * 2, # Group C still insufficient
        'Value': list(np.random.normal(loc=5, scale=1, size=n_samples_good)) +  # Group A: Normal
                 list(np.random.uniform(low=0, high=10, size=n_samples_good)) + # Group B: Uniform (Not Normal)
                 [1, 1]                                                        # Group C
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_with_hue():
    """DataFrame with two grouping variables (x, hue)"""
    np.random.seed(123) # for reproducibility
    groups = ['X', 'Y']
    conditions = ['Control', 'Treatment']
    data_list = []
    # X, Control: Normal
    data_list.extend([{'Category': 'X', 'Condition': 'Control', 'Score': v} for v in np.random.normal(10, 2, 20)])
    # X, Treatment: Exponential (Not Normal)
    data_list.extend([{'Category': 'X', 'Condition': 'Treatment', 'Score': v} for v in np.random.exponential(5, 20)])
    # Y, Control: Normal
    data_list.extend([{'Category': 'Y', 'Condition': 'Control', 'Score': v} for v in np.random.normal(15, 3, 25)])
    # Y, Treatment: Insufficient Data
    data_list.extend([{'Category': 'Y', 'Condition': 'Treatment', 'Score': v} for v in [20, 22]])
    # Y, Extra (Not in default hue_order) - Normal
    data_list.extend([{'Category': 'Y', 'Condition': 'Extra', 'Score': v} for v in np.random.normal(18, 1, 15)])

    return pd.DataFrame(data_list)

# --- Test Functions ---

def test_no_hue_basic(sample_data_no_hue):
    """Test basic functionality without hue."""
    results = normality_shapiro_wilk_test(data=sample_data_no_hue, x='Group', y='Value')

    assert isinstance(results, dict)
    assert 'A' in results
    assert 'B' in results
    assert 'C' in results

    # Group A: Expected Normal (high p-value)
    assert results['A']['n_samples'] == sample_data_no_hue.loc[sample_data_no_hue['Group'] == 'A', 'Value'].dropna().shape[0]
    assert results['A']['p_value'] > 0.05 # Probabilistic check

    # Group B: Expected Not Normal (low p-value)
    assert results['B']['n_samples'] == sample_data_no_hue.loc[sample_data_no_hue['Group'] == 'B', 'Value'].dropna().shape[0]
    assert results['B']['p_value'] < 0.05 # Probabilistic check

    # Group C: Insufficient samples
    assert results['C']['n_samples'] == 2
    assert math.isnan(results['C']['statistic'])
    assert math.isnan(results['C']['p_value'])

def test_with_hue_basic(sample_data_with_hue):
    """Test basic functionality with hue."""
    results = normality_shapiro_wilk_test(data=sample_data_with_hue, x='Category', y='Score', hue='Condition')

    assert isinstance(results, dict)
    assert ('X', 'Control') in results
    assert ('X', 'Treatment') in results
    assert ('Y', 'Control') in results
    assert ('Y', 'Treatment') in results
    assert ('Y', 'Extra') in results # Included because hue_order=None

    # Group ('X', 'Control'): Expected Normal
    assert results[('X', 'Control')]['n_samples'] == 20
    assert results[('X', 'Control')]['p_value'] > 0.05

    # Group ('X', 'Treatment'): Expected Not Normal (Exponential)
    assert results[('X', 'Treatment')]['n_samples'] == 20
    assert results[('X', 'Treatment')]['p_value'] < 0.05

    # Group ('Y', 'Control'): Expected Normal
    assert results[('Y', 'Control')]['n_samples'] == 25
    assert results[('Y', 'Control')]['p_value'] > 0.05

    # Group ('Y', 'Treatment'): Insufficient samples
    assert results[('Y', 'Treatment')]['n_samples'] == 2
    assert math.isnan(results[('Y', 'Treatment')]['statistic'])
    assert math.isnan(results[('Y', 'Treatment')]['p_value'])

    # Group ('Y', 'Extra'): Expected Normal
    assert results[('Y', 'Extra')]['n_samples'] == 15
    assert results[('Y', 'Extra')]['p_value'] > 0.05


def test_order_respected(sample_data_no_hue):
    """Test if the 'order' argument correctly filters and orders (implicitly)."""
    custom_order = ['B', 'A'] # Only test B and A, in this order
    results = normality_shapiro_wilk_test(data=sample_data_no_hue, x='Group', y='Value', order=custom_order)

    assert list(results.keys()) == custom_order # Check keys are present and potentially ordered
    assert 'C' not in results # Group C was excluded by 'order'

    # Check content for B
    assert results['B']['n_samples'] == sample_data_no_hue.loc[sample_data_no_hue['Group'] == 'B', 'Value'].dropna().shape[0]
    assert results['B']['p_value'] < 0.05

    # Check content for A
    assert results['A']['n_samples'] == sample_data_no_hue.loc[sample_data_no_hue['Group'] == 'A', 'Value'].dropna().shape[0]
    assert results['A']['p_value'] > 0.05

def test_hue_order_respected(sample_data_with_hue):
    """Test if 'hue_order' correctly filters and orders (implicitly)."""
    custom_x_order = ['Y', 'X']
    custom_hue_order = ['Treatment', 'Control'] # Exclude 'Extra'
    results = normality_shapiro_wilk_test(
        data=sample_data_with_hue,
        x='Category',
        y='Score',
        hue='Condition',
        order=custom_x_order,
        hue_order=custom_hue_order
    )

    expected_keys = [
        ('Y', 'Treatment'), ('Y', 'Control'),
        ('X', 'Treatment'), ('X', 'Control')
    ]
    assert list(results.keys()) == expected_keys # Check keys are present and potentially ordered
    assert ('Y', 'Extra') not in results # Excluded by hue_order

    # Check content for ('Y', 'Treatment')
    assert results[('Y', 'Treatment')]['n_samples'] == 2
    assert math.isnan(results[('Y', 'Treatment')]['p_value'])

    # Check content for ('X', 'Control')
    assert results[('X', 'Control')]['n_samples'] == 20
    assert results[('X', 'Control')]['p_value'] > 0.05


def test_min_samples_check():
    """Test that the minimum sample check works correctly."""
    df = pd.DataFrame({
        'Group': ['A']*2 + ['B']*3,
        'Value': [1,2] + [10,11,12]
    })
    # Test with default min_samples=3
    results_default = normality_shapiro_wilk_test(data=df, x='Group', y='Value')
    assert math.isnan(results_default['A']['p_value']) # n=2 < 3
    assert not math.isnan(results_default['B']['p_value']) # n=3 >= 3

    # Test with min_samples=4
    results_4 = normality_shapiro_wilk_test(data=df, x='Group', y='Value', min_samples=4)
    assert math.isnan(results_4['A']['p_value']) # n=2 < 4
    assert math.isnan(results_4['B']['p_value']) # n=3 < 4


def test_missing_column_errors(sample_data_no_hue):
    """Test that ValueErrors are raised for non-existent columns."""
    with pytest.raises(ValueError, match="Column 'NonExistentX' not found"):
        normality_shapiro_wilk_test(data=sample_data_no_hue, x='NonExistentX', y='Value')

    with pytest.raises(ValueError, match="Column 'NonExistentY' not found"):
        normality_shapiro_wilk_test(data=sample_data_no_hue, x='Group', y='NonExistentY')

    with pytest.raises(ValueError, match="Column 'NonExistentHue' not found"):
        normality_shapiro_wilk_test(data=sample_data_no_hue, x='Group', y='Value', hue='NonExistentHue')

def test_min_samples_value_error(sample_data_no_hue):
    """Test that ValueError is raised if min_samples < 3."""
    with pytest.raises(ValueError, match="Shapiro-Wilk test requires at least 3 samples"):
        normality_shapiro_wilk_test(data=sample_data_no_hue, x='Group', y='Value', min_samples=2)

def test_empty_dataframe():
    """Test with an empty DataFrame."""
    df = pd.DataFrame({'Group': [], 'Value': [], 'Condition': []})
    results = normality_shapiro_wilk_test(data=df, x='Group', y='Value')
    assert results == {} # Expect an empty dictionary

    results_hue = normality_shapiro_wilk_test(data=df, x='Group', y='Value', hue='Condition')
    assert results_hue == {}

def test_nan_handling(sample_data_no_hue):
    """Test that NaN values in y are handled correctly."""
    # Add some NaNs
    sample_data_no_hue.loc[0, 'Value'] = np.nan
    sample_data_no_hue.loc[30, 'Value'] = np.nan # First value of Group B
    sample_data_no_hue.loc[2, 'Value'] = np.nan

    results = normality_shapiro_wilk_test(data=sample_data_no_hue, x='Group', y='Value')

    # Group A: Expected Normal (high p-value) - n should be 28 now
    assert results['A']['n_samples'] == sample_data_no_hue.loc[sample_data_no_hue['Group'] == 'A', 'Value'].dropna().shape[0]
    assert results['A']['p_value'] > 0.05 # Still likely normal

    # Group B: Expected Not Normal (low p-value) - n should be 29 now
    assert results['B']['n_samples'] == sample_data_no_hue.loc[sample_data_no_hue['Group'] == 'B', 'Value'].dropna().shape[0]
    assert results['B']['p_value'] < 0.05 # Still likely uniform

    # Group C: Insufficient samples - n is still 2
    assert results['C']['n_samples'] == 2
    assert math.isnan(results['C']['statistic'])