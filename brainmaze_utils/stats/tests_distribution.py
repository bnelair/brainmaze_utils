
import pandas as pd
import numpy as np
from scipy import stats # Use scipy.stats for f distribution
from typing import Optional, List, Dict, Any, Tuple, Union
import math


def normality_shapiro_wilk_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    min_samples: int = 3
) -> Dict[Union[str, Tuple[str, str]], Dict[str, Any]]:
    """
    Assesses normality for groups using the Shapiro-Wilk test.

    Mimics the grouping logic of seaborn's boxplot (x, hue, order, hue_order)
    and performs the Shapiro-Wilk test on the 'y' variable for each group.

    Parameters
    ----------
    data : pd.DataFrame
        The input tabular data.
    x : str
        Name of the variable in `data` for the main grouping category (x-axis).
    y : str
        Name of the variable in `data` containing the numerical data to test.
    hue : str, optional
        Name of the variable in `data` for the secondary grouping category.
        If None, only groups by 'x'. Defaults to None.
    order : list[str], optional
        Specific order to process categories in 'x'. If None, uses sorted unique
        values from `data[x]`. Defaults to None.
    hue_order : list[str], optional
        Specific order to process categories in 'hue'. Only used if `hue` is
        not None. If None, uses sorted unique values from `data[hue]`.
        Defaults to None.
    min_samples : int, optional
        The minimum number of non-NaN samples required in a group to perform
        the Shapiro-Wilk test. SciPy requires at least 3. Defaults to 3.

    Returns
    -------
    dict
        A dictionary where keys represent the groups and values are dictionaries
        containing the Shapiro-Wilk test results.
        - If `hue` is None, keys are strings from the 'x' column.
        - If `hue` is provided, keys are tuples: (x_category, hue_category).
        Each value dictionary contains:
            - 'n_samples': (int) Number of non-NaN samples in the group.
            - 'statistic': (float) The Shapiro-Wilk test statistic, or np.nan
                           if n_samples < min_samples.
            - 'p_value': (float) The p-value for the test, or np.nan
                         if n_samples < min_samples.

    Raises
    ------
    ValueError
        If specified x, y, or hue columns are not in the DataFrame.
        If min_samples < 3.
    KeyError
        If categories specified in order/hue_order are used for filtering
        but do not exist in the data (though typically pandas handles this
        gracefully by returning an empty slice).

    Notes
    -----
    - The null hypothesis of the Shapiro-Wilk test is that the data was
      drawn from a normal distribution. A small p-value (e.g., < 0.05)
      suggests rejecting the null hypothesis (i.e., data is likely not normal).
    - NaN values in the 'y' column are automatically dropped for each group
      before testing.
    """
    # --- Input Validation ---
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")
    if y not in data.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame.")
    if hue is not None and hue not in data.columns:
        raise ValueError(f"Column '{hue}' not found in DataFrame.")
    if min_samples < 3:
        raise ValueError("Shapiro-Wilk test requires at least 3 samples (min_samples >= 3).")

    # --- Determine Iteration Order ---
    if order is None:
        try:
            # Attempt numerical sort if possible, otherwise string sort
            x_categories = sorted(data[x].dropna().unique(), key=lambda v: float(v))
        except (ValueError, TypeError):
            x_categories = sorted(data[x].dropna().unique())
    else:
        x_categories = order

    hue_categories = None
    if hue is not None:
        if hue_order is None:
             try:
                 # Attempt numerical sort if possible, otherwise string sort
                 hue_categories = sorted(data[hue].dropna().unique(), key=lambda v: float(v))
             except (ValueError, TypeError):
                 hue_categories = sorted(data[hue].dropna().unique())
        else:
            hue_categories = hue_order

    # --- Perform Tests ---
    results = {}
    for x_cat in x_categories:
        # Filter data for the current x category
        data_x_filtered = data[data[x] == x_cat]
        if data_x_filtered.empty:
            continue # Skip if order contains categories not in data

        if hue is None or hue_categories is None:
            # No hue grouping
            series = data_x_filtered[y].dropna()
            n = len(series)
            stat, p_val = np.nan, np.nan # Default if not enough samples
            if n >= min_samples:
                try:
                    stat, p_val = stats.shapiro(series)
                except Exception as e:
                    # Handle potential errors within shapiro, though unlikely with size check
                    print(f"Warning: Shapiro test failed for group '{x_cat}': {e}")


            results[x_cat] = {
                'n_samples': n,
                'statistic': stat,
                'p_value': p_val
            }
        else:
            # Hue grouping
            for h_cat in hue_categories:
                group_key = (x_cat, h_cat)
                # Filter further by hue category
                data_h_filtered = data_x_filtered[data_x_filtered[hue] == h_cat]
                if data_h_filtered.empty and h_cat not in data_x_filtered[hue].unique():
                     # Don't create entry if combination doesn't exist AT ALL in the data for this x_cat
                     # Useful if hue_order contains values not present for a specific x_cat
                     continue

                series = data_h_filtered[y].dropna()
                n = len(series)
                stat, p_val = np.nan, np.nan # Default if not enough samples
                if n >= min_samples:
                     try:
                         stat, p_val = stats.shapiro(series)
                     except Exception as e:
                         print(f"Warning: Shapiro test failed for group {group_key}: {e}")

                results[group_key] = {
                    'n_samples': n,
                    'statistic': stat,
                    'p_value': p_val
                }

    return results


def variance_homogeneity_f_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    min_samples: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Assesses homogeneity of variances between TWO groups using the F-test.

    This function expects the parameters (x, hue, order, hue_order) to define
    exactly two groups present in the data. It calculates the F-statistic
    and p-value to test if the variances of the 'y' variable are equal
    between these two specific groups.

    Parameters
    ----------
    data : pd.DataFrame
        The input tabular data.
    x : str
        Name of the variable in `data` for the main grouping category.
    y : str
        Name of the variable in `data` containing the numerical data.
    hue : str, optional
        Name of the variable in `data` for the secondary grouping category.
        If None, expects 'order' to contain exactly two category names from 'x'.
        Defaults to None.
    order : list[str], optional
        Specific order/selection for categories in 'x'.
        - If `hue` is None, this list MUST contain exactly two category names.
        - If `hue` is not None, this list MUST contain exactly one category name.
        Defaults to None (which will likely cause an error unless data[x] only has 2 unique values).
    hue_order : list[str], optional
        Specific order/selection for categories in 'hue'. Only used if `hue` is not None.
        If `hue` is not None, this list MUST contain exactly two category names.
        Defaults to None (which will likely cause an error unless relevant data[hue] only has 2 unique values).
    min_samples : int, optional
        The minimum number of non-NaN samples required in EACH of the two groups
        to perform the F-test. Variance calculation requires at least 2.
        Defaults to 2.

    Returns
    -------
    dict or None
        A dictionary containing the F-test results if exactly two valid groups
        are specified and meet the minimum sample requirement. Otherwise, returns None.
        The dictionary contains:
            - 'group1_id': Identifier of the first group (str or tuple).
            - 'group2_id': Identifier of the second group (str or tuple).
            - 'n1': (int) Number of non-NaN samples in group 1.
            - 'n2': (int) Number of non-NaN samples in group 2.
            - 'variance1': (float) Sample variance of group 1 (ddof=1).
            - 'variance2': (float) Sample variance of group 2 (ddof=1).
            - 'f_statistic': (float) F = larger_variance / smaller_variance. NaN if a variance is 0 or negative.
            - 'df_num': (int) Numerator degrees of freedom (n - 1 for group with larger variance).
            - 'df_den': (int) Denominator degrees of freedom (n - 1 for group with smaller variance).
            - 'p_value': (float) Two-tailed p-value for the test. NaN if F is NaN.

    Raises
    ------
    ValueError
        If specified x, y, or hue columns are not in the DataFrame.
        If min_samples < 2.
        If the combination of x, hue, order, hue_order does not specify
        exactly two groups found within the data.

    Notes
    -----
    - The null hypothesis (H0) of the F-test for variances is that the variances
      of the two groups are equal. A small p-value (e.g., < 0.05) suggests
      rejecting H0 (i.e., variances are likely different).
    - NaN values in the 'y' column are automatically dropped for each group.
    - The F-test assumes the data within each group is normally distributed.
      Violations of normality can affect the reliability of the F-test result.
      It's often recommended to check normality first (e.g., using Shapiro-Wilk).
    """
    # --- Input Validation ---
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")
    if y not in data.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame.")
    if hue is not None and hue not in data.columns:
        raise ValueError(f"Column '{hue}' not found in DataFrame.")
    if min_samples < 2:
        raise ValueError("F-test requires at least 2 samples per group (min_samples >= 2).")

    # --- Identify the two groups ---
    group_ids = []
    group_data_series = []

    if hue is None:
        if order is None or len(order) != 2:
            raise ValueError("If 'hue' is None, 'order' must be a list containing exactly two category names from column '{}'.".format(x))
        group_ids = order
        for group_id in group_ids:
            series = data.loc[data[x] == group_id, y].dropna()
            group_data_series.append(series)
    else:
        if order is None or len(order) != 1:
             raise ValueError("If 'hue' is not None, 'order' must be a list containing exactly one category name from column '{}'.".format(x))
        if hue_order is None or len(hue_order) != 2:
             raise ValueError("If 'hue' is not None, 'hue_order' must be a list containing exactly two category names from column '{}'.".format(hue))

        x_cat = order[0]
        group_ids = [(x_cat, h_cat) for h_cat in hue_order]
        for group_id_tuple in group_ids:
            _, h_cat = group_id_tuple
            series = data.loc[(data[x] == x_cat) & (data[hue] == h_cat), y].dropna()
            group_data_series.append(series)

    # --- Validate Group Data ---
    if len(group_data_series) != 2:
         # This case should ideally be caught by the order/hue_order checks, but as a safeguard:
         print(f"Warning: Could not isolate exactly two data series for groups {group_ids}. Check data and parameters.")
         return None

    series1, series2 = group_data_series
    n1, n2 = len(series1), len(series2)
    group1_id, group2_id = group_ids

    if n1 < min_samples or n2 < min_samples:
        print(f"Warning: Insufficient samples for F-test. Group '{group1_id}' (n={n1}), Group '{group2_id}' (n={n2}). Minimum required: {min_samples}.")
        # Return None or a structure indicating failure? Let's return None.
        return None
        # Alternative: Return results with NaNs
        # return {
        #      'group1_id': group1_id, 'group2_id': group2_id,
        #      'n1': n1, 'n2': n2, 'variance1': np.nan, 'variance2': np.nan,
        #      'f_statistic': np.nan, 'df_num': np.nan, 'df_den': np.nan, 'p_value': np.nan
        # }


    # --- Calculate Variances and F-statistic ---
    var1 = np.var(series1, ddof=1)
    var2 = np.var(series2, ddof=1)

    # Handle zero variance case
    if var1 <= 0 or var2 <= 0:
         # If one is zero and other positive, they are different. If both zero, they are equal.
         # F-statistic calculation fails if denominator is 0.
         # SciPy's f distribution requires F > 0, df > 0.
         is_equal = (var1 == var2) # True if both are zero, False otherwise
         return {
             'group1_id': group1_id, 'group2_id': group2_id,
             'n1': n1, 'n2': n2, 'variance1': var1, 'variance2': var2,
             'f_statistic': np.nan, # Cannot compute F or use distribution
             'df_num': n1 - 1 if var1 >= var2 else n2 - 1, # Assign based on convention
             'df_den': n2 - 1 if var1 >= var2 else n1 - 1, # Assign based on convention
             'p_value': 1.0 if is_equal else 0.0 # p=1 if equal (both 0), p=0 if unequal (one is 0)
         }

    # Ensure F >= 1 by putting larger variance in numerator
    if var1 >= var2:
        f_stat = var1 / var2
        df_num = n1 - 1
        df_den = n2 - 1
    else:
        f_stat = var2 / var1
        df_num = n2 - 1
        df_den = n1 - 1

    # --- Calculate p-value (two-tailed) ---
    # sf = 1 - cdf. We want P(F > f_stat) for the right tail.
    # The p-value for a two-tailed test is 2 * min(one-tailed p-value, 1 - one-tailed p-value)
    # Or simply 2 * P(F > f_stat) assuming f_stat >= 1
    p_value = stats.f.sf(f_stat, df_num, df_den) * 2

    # Clamp p-value to be at most 1.0 (can slightly exceed due to float precision)
    p_value = min(p_value, 1.0)

    results = {
        'group1_id': group1_id,
        'group2_id': group2_id,
        'n1': n1,
        'n2': n2,
        'variance1': var1,
        'variance2': var2,
        'f_statistic': f_stat,
        'df_num': df_num,
        'df_den': df_den,
        'p_value': p_value
    }

    return results


def assess_homogeneity_brown_forsythe(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    min_samples: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Assesses homogeneity of variances across two or more groups using the
    Brown-Forsythe test (Levene test centered at the median).

    Mimics the grouping logic of seaborn's boxplot (x, hue, order, hue_order)
    and performs the test on the 'y' variable across all specified groups
    that meet the minimum sample size requirement.

    Parameters
    ----------
    data : pd.DataFrame
        The input tabular data.
    x : str
        Name of the variable in `data` for the main grouping category (x-axis).
    y : str
        Name of the variable in `data` containing the numerical data to test.
    hue : str, optional
        Name of the variable in `data` for the secondary grouping category.
        If None, groups only by 'x'. Defaults to None.
    order : list[str], optional
        Specific order/selection for categories in 'x'. If None, uses sorted unique
        values from `data[x]`. Defaults to None.
    hue_order : list[str], optional
        Specific order/selection for categories in 'hue'. Only used if `hue` is
        not None. If None, uses sorted unique values from `data[hue]`.
        Defaults to None.
    min_samples : int, optional
        The minimum number of non-NaN samples required in a group for it to be
        included in the Brown-Forsythe test. Defaults to 3.

    Returns
    -------
    dict or None
        A dictionary containing the Brown-Forsythe test results if at least two
        valid groups are found and meet the minimum sample requirement.
        Otherwise, returns None.
        The dictionary contains:
            - 'groups_tested': (list) Identifiers of groups included in the test
                               (str or tuple).
            - 'n_groups_tested': (int) Number of groups included in the test.
            - 'statistic': (float) The Brown-Forsythe test statistic.
            - 'p_value': (float) The p-value for the test.

    Raises
    ------
    ValueError
        If specified x, y, or hue columns are not in the DataFrame.
        If min_samples is less than necessary (e.g., < 2).
    KeyError
        If categories specified in order/hue_order are used for filtering
        but do not exist in the data.

    Notes
    -----
    - This function uses `scipy.stats.levene` with `center='median'`.
    - The null hypothesis (H0) is that all groups have equal variances.
      A small p-value (e.g., < 0.05) suggests rejecting H0.
    - NaN values in the 'y' column are automatically dropped for each group.
    """
    # --- Input Validation ---
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")
    if y not in data.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame.")
    if hue is not None and hue not in data.columns:
        raise ValueError(f"Column '{hue}' not found in DataFrame.")
    if min_samples < 2:
         raise ValueError("min_samples must be >= 2 for variance calculation within groups.")


    # --- Determine Iteration Order and Collect Group Data ---
    if order is None:
        try:
            x_categories = sorted(data[x].dropna().unique(), key=lambda v: float(v))
        except (ValueError, TypeError):
            x_categories = sorted(data[x].dropna().unique())
    else:
        x_categories = order

    hue_categories = None
    if hue is not None:
        if hue_order is None:
             try:
                 hue_categories = sorted(data[hue].dropna().unique(), key=lambda v: float(v))
             except (ValueError, TypeError):
                 hue_categories = sorted(data[hue].dropna().unique())
        else:
            hue_categories = hue_order

    valid_group_data = []
    valid_group_ids = []

    for x_cat in x_categories:
        data_x_filtered = data[data[x] == x_cat]
        if data_x_filtered.empty:
            continue

        if hue is None or hue_categories is None:
            # No hue grouping
            group_id = x_cat
            series = data_x_filtered[y].dropna()
            n = len(series)
            if n >= min_samples:
                valid_group_ids.append(group_id)
                valid_group_data.append(series.to_numpy())
        else:
            # Hue grouping
            for h_cat in hue_categories:
                group_id = (x_cat, h_cat)
                data_h_filtered = data_x_filtered[data_x_filtered[hue] == h_cat]
                # Check if this specific combination exists before proceeding
                if data_h_filtered.empty:
                     if h_cat not in data_x_filtered[hue].unique():
                         continue
                     else:
                         series = pd.Series(dtype=float) # Empty series
                else:
                    series = data_h_filtered[y].dropna()

                n = len(series)
                if n >= min_samples:
                    valid_group_ids.append(group_id)
                    valid_group_data.append(series.to_numpy())

    # --- Perform Test ---
    if len(valid_group_data) < 2:
        print(f"Warning: Fewer than two groups met the minimum sample requirement ({min_samples}). Brown-Forsythe test cannot be performed.")
        return None

    try:
        # Need at least two arrays to unpack for levene
        if len(valid_group_data) >= 2:
             statistic, p_value = stats.levene(*valid_group_data, center='median')
        else: # Should have been caught above, but safeguard
             return None
    except Exception as e:
        print(f"Error during Brown-Forsythe test execution: {e}")
        return None

    results = {
        'groups_tested': valid_group_ids,
        'n_groups_tested': len(valid_group_ids),
        'statistic': statistic,
        'p_value': p_value
    }

    return results


