import pytest
import pandas as pd
import numpy as np

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





