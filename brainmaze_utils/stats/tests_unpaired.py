
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Dict, Any, Tuple, Union
import math

# In e.g., comparison_tests.py or normality.py
# Ensure imports for pandas, numpy, typing, math, scipy.stats are present
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Dict, Any, Tuple, Union
import math

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Dict, Any, Tuple, Union
import math
import warnings # To warn about normality for ANOVA

import scikit_posthocs as sp

def compare_two_groups_unpaired(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    alpha: float = 0.05,
    min_samples_normality: int = 3,
    min_samples_variance: int = 2 # Min samples needed for variance calc
) -> Optional[Dict[str, Any]]:
    """
    Compares two independent groups using t-test or Mann-Whitney U test.

    Automatically selects the test based on Shapiro-Wilk normality (p > alpha)
    and F-test for homogeneity of variances (p > alpha). Expects parameters
    to define exactly two groups.

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
    order : list[str], optional
        Specific order/selection for categories in 'x'. MUST contain exactly two names if hue is None.
        MUST contain exactly one name if hue is not None.
    hue_order : list[str], optional
        Specific order/selection for categories in 'hue'. MUST contain exactly two names if hue is not None.
    alpha : float, optional
        Significance level for Shapiro-Wilk and F-test. Defaults to 0.05.
    min_samples_normality : int, optional
        Minimum non-NaN samples required in EACH group for Shapiro-Wilk. Defaults to 3.
    min_samples_variance : int, optional
        Minimum non-NaN samples required in EACH group for F-test (variance calc). Defaults to 2.

    Returns
    -------
    dict or None
        A dictionary containing the preliminary test results and the final comparison results.
        Returns None if prerequisites (exactly 2 groups, min samples) are not met.
        Dictionary includes:
            - 'group1_id', 'group2_id': Identifiers of the groups.
            - 'n1', 'n2': Number of non-NaN samples per group.
            - 'normality_p_value1', 'normality_p_value2': Shapiro-Wilk p-values.
            - 'variance_test_used': Name of the variance test ('F-test' or 'N/A').
            - 'variance1', 'variance2': Calculated sample variances.
            - 'f_statistic': F-statistic from variance test (NaN if not applicable).
            - 'variance_p_value': P-value from variance test (NaN if not applicable).
            - 'test_used': Comparison test performed ('Student t-test', 'Welch t-test',
                           'Mann-Whitney U', or 'N/A').
            - 'statistic': Comparison test statistic.
            - 'p_value': Comparison test p-value.

    Raises
    ------
    ValueError
        If input parameters are invalid (missing columns, incorrect number of groups specified, etc.).
    """
    # --- Input Validation ---
    if x not in data.columns: raise ValueError(f"Column '{x}' not found.")
    if y not in data.columns: raise ValueError(f"Column '{y}' not found.")
    if hue is not None and hue not in data.columns: raise ValueError(f"Column '{hue}' not found.")
    if min_samples_normality < 3: raise ValueError("min_samples_normality must be >= 3.")
    if min_samples_variance < 2: raise ValueError("min_samples_variance must be >= 2.")

    min_samples_needed = max(min_samples_normality, min_samples_variance)

    # --- Identify the two groups ---
    group_ids = []
    group_data_series = []
    if hue is None:
        if order is None or len(order) != 2: raise ValueError(f"If 'hue' is None, 'order' must be a list containing exactly two category names from column '{x}'.")
        group_ids = order
        for group_id in group_ids:
            if group_id not in data[x].unique(): raise ValueError(f"Group '{group_id}' in 'order' not found in column '{x}'.")
            series = data.loc[data[x] == group_id, y].dropna()
            group_data_series.append(series)
    else:
        if order is None or len(order) != 1: raise ValueError(f"If 'hue' is not None, 'order' must contain exactly one category name from column '{x}'.")
        if hue_order is None or len(hue_order) != 2: raise ValueError(f"If 'hue' is not None, 'hue_order' must contain exactly two category names from column '{hue}'.")
        x_cat = order[0]
        if x_cat not in data[x].unique(): raise ValueError(f"Group '{x_cat}' in 'order' not found in column '{x}'.")
        group_ids = [(x_cat, h_cat) for h_cat in hue_order]
        for group_id_tuple in group_ids:
            _, h_cat = group_id_tuple
            # Check if hue category exists within the selected x category
            if h_cat not in data.loc[data[x] == x_cat, hue].unique():
                raise ValueError(f"Group '{group_id_tuple}' specified by 'hue_order' not found for x='{x_cat}'.")
            series = data.loc[(data[x] == x_cat) & (data[hue] == h_cat), y].dropna()
            group_data_series.append(series)

    if len(group_data_series) != 2: return None # Should be caught earlier

    series1, series2 = group_data_series
    n1, n2 = len(series1), len(series2)
    group1_id, group2_id = group_ids

    # Check minimum samples *before* running tests
    if n1 < min_samples_needed or n2 < min_samples_needed:
        print(f"Warning: Insufficient samples for testing. Group '{group1_id}' (n={n1}), Group '{group2_id}' (n={n2}). Minimum required: {min_samples_needed}.")
        # Return partial info indicating failure point
        return {
            'group1_id': group1_id, 'group2_id': group2_id, 'n1': n1, 'n2': n2,
            'normality_p_value1': np.nan, 'normality_p_value2': np.nan,
            'variance_test_used': 'N/A', 'variance1': np.nan, 'variance2': np.nan,
            'f_statistic': np.nan, 'variance_p_value': np.nan,
            'test_used': 'N/A (Insufficient Samples)',
            'statistic': np.nan, 'p_value': np.nan
        }


    # --- Perform Normality Tests ---
    shapiro_p1, shapiro_p2 = np.nan, np.nan
    try: _, shapiro_p1 = stats.shapiro(series1)
    except Exception as e: print(f"Warning: Shapiro test failed group 1: {e}")
    try: _, shapiro_p2 = stats.shapiro(series2)
    except Exception as e: print(f"Warning: Shapiro test failed group 2: {e}")

    # --- Perform F-test for Variances ---
    f_stat, f_p_value = np.nan, np.nan
    var_test_name = "N/A"
    var1, var2 = np.nan, np.nan
    homogeneity_passed = False # Default assumption

    if n1 >= min_samples_variance and n2 >= min_samples_variance: # Check specific min for variance test
        var1 = np.var(series1, ddof=1)
        var2 = np.var(series2, ddof=1)
        var_test_name = "F-test"

        if var1 > 0 and var2 > 0:
            if var1 >= var2: f_stat, df_num, df_den = var1 / var2, n1 - 1, n2 - 1
            else: f_stat, df_num, df_den = var2 / var1, n2 - 1, n1 - 1
            f_p_value = min(stats.f.sf(f_stat, df_num, df_den) * 2, 1.0)
        elif var1 == var2: f_p_value = 1.0 # Both zero or identical non-zero
        else: f_p_value = 0.0 # One zero, one positive

        if not math.isnan(f_p_value):
             homogeneity_passed = f_p_value > alpha
    else:
        # Can't run variance test, try to report calculated variances if possible
        if n1 >= 2: var1 = np.var(series1, ddof=1)
        if n2 >= 2: var2 = np.var(series2, ddof=1)


    # --- Choose and Perform Comparison Test ---
    test_name = "N/A"
    comp_stat = np.nan
    comp_p_value = np.nan
    normality_passed = False

    if not (math.isnan(shapiro_p1) or math.isnan(shapiro_p2)):
        normality_passed = shapiro_p1 > alpha and shapiro_p2 > alpha

        if normality_passed:
            # Both Normal: Choose t-test based on F-test result (homogeneity_passed)
            if math.isnan(f_p_value): # F-test couldn't run
                 test_name = "N/A (Variance Test Failed)"
            elif homogeneity_passed:
                 test_name = "Student t-test"
                 try: comp_stat, comp_p_value = stats.ttest_ind(series1, series2, equal_var=True, alternative='two-sided', nan_policy='omit')
                 except Exception as e: print(f"Error: Student t-test: {e}"); test_name = "Student t-test Failed"
            else: # Not homogeneous
                 test_name = "Welch t-test"
                 try: comp_stat, comp_p_value = stats.ttest_ind(series1, series2, equal_var=False, alternative='two-sided', nan_policy='omit')
                 except Exception as e: print(f"Error: Welch t-test: {e}"); test_name = "Welch t-test Failed"
        else:
            # At least one Not Normal: Use Mann-Whitney U
            test_name = "Mann-Whitney U"
            if n1 < 7 or n2 < 7: print(f"Warning: Sample size(s) ({n1}, {n2}) may be small for Mann-Whitney U.")
            try: comp_stat, comp_p_value = stats.mannwhitneyu(series1, series2, alternative='two-sided')
            except Exception as e: print(f"Error: Mann-Whitney U: {e}"); test_name = "Mann-Whitney U Failed"
    else:
        test_name = "N/A (Normality Failed)"

    # --- Compile Results ---
    results = {
        'group1_id': group1_id, 'group2_id': group2_id,
        'n1': n1, 'n2': n2,
        'normality_p_value1': shapiro_p1, 'normality_p_value2': shapiro_p2,
        'variance_test_used': var_test_name,
        'variance1': var1, 'variance2': var2,
        'f_statistic': f_stat, 'variance_p_value': f_p_value,
        'test_used': test_name,
        'statistic': comp_stat, 'p_value': comp_p_value
    }
    return results

def compare_multi_groups(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    alpha: float = 0.05,
    min_samples: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Compares means/distributions across three or more independent groups.

    Selects between ANOVA + Bonferroni post-hoc (if Brown-Forsythe p > alpha)
    and Kruskal-Wallis + Dunn's post-hoc (if Brown-Forsythe p <= alpha).
    Uses the specified alpha for both homogeneity and overall comparison tests.

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
    order : list[str], optional
        Specific order/selection for categories in 'x'. If None, uses all unique
        sorted values from `data[x]`.
    hue_order : list[str], optional
        Specific order/selection for categories in 'hue'. Only used if `hue` is not None.
        If None, uses all unique sorted values from `data[hue]`.
    alpha : float, optional
        Significance level for Brown-Forsythe, ANOVA/Kruskal-Wallis, and post-hoc tests.
        Defaults to 0.05.
    min_samples : int, optional
        Min non-NaN samples per group to be included. Defaults to 3.

    Returns
    -------
    dict or None
        A dictionary containing preliminary test results, overall comparison results,
        and post-hoc results (if applicable). Returns None if fewer than 3 valid
        groups are found or if a critical test fails.
        Dictionary includes:
            - 'groups_tested': (list) Identifiers of groups included (str or tuple).
            - 'n_groups_tested': (int) Number of groups included.
            - 'homogeneity_test': (str) 'Brown-Forsythe'.
            - 'homogeneity_stat': (float) Brown-Forsythe statistic.
            - 'homogeneity_p_value': (float) Brown-Forsythe p-value.
            - 'overall_comparison_test': (str) 'ANOVA' or 'Kruskal-Wallis'.
            - 'overall_stat': (float) F-statistic or H-statistic.
            - 'overall_p_value': (float) P-value for the overall comparison.
            - 'posthoc_test': (str) 'Pairwise t-test (Bonferroni)', 'Dunn (Bonferroni)', or 'N/A'.
            - 'posthoc_results': (pd.DataFrame or None) DataFrame of pairwise p-values
                                 from scikit-posthocs, or None if no post-hoc was run.

    Raises
    ------
    ValueError
        If input parameters are invalid or fewer than 3 groups are specified/found.
    ImportError
        If scikit-posthocs is not installed.

    Notes
    -----
    - Requires `scikit-posthocs` for post-hoc tests.
    - Assumes normality for ANOVA, although the choice here is driven by homogeneity test per user spec.
    - NaN values in the 'y' column are automatically dropped.
    """
    # --- Input Validation ---
    if x not in data.columns: raise ValueError(f"Column '{x}' not found.")
    if y not in data.columns: raise ValueError(f"Column '{y}' not found.")
    if hue is not None and hue not in data.columns: raise ValueError(f"Column '{hue}' not found.")
    if min_samples < 2: raise ValueError("min_samples must be >= 2.")

    # --- Determine Iteration Order and Collect Group Data ---
    # (Logic for identifying x_categories and hue_categories remains the same)
    # ... (omitted for brevity - see assess_homogeneity_brown_forsythe) ...
    if order is None:
        try: x_categories = sorted(data[x].dropna().unique(), key=lambda v: float(v))
        except (ValueError, TypeError): x_categories = sorted(data[x].dropna().unique())
    else: x_categories = order
    hue_categories = None
    if hue is not None:
        if hue_order is None:
             try: hue_categories = sorted(data[hue].dropna().unique(), key=lambda v: float(v))
             except (ValueError, TypeError): hue_categories = sorted(data[hue].dropna().unique())
        else: hue_categories = hue_order

    valid_group_data_map = {} # Store series mapped to group_id
    valid_group_ids = []

    group_col_name = "_temp_group_id_" # For post-hoc processing

    # Create a combined group identifier if hue is used
    if hue:
        # Ensure components are string for reliable concatenation
        data[group_col_name] = data[x].astype(str) + "__" + data[hue].astype(str)
    else:
        data[group_col_name] = data[x] # Use x directly if no hue

    # Collect valid data
    for x_cat in x_categories:
        data_x_filtered = data[data[x] == x_cat]
        if data_x_filtered.empty: continue

        if hue is None or hue_categories is None:
            group_id = x_cat
            series = data_x_filtered[y].dropna()
            n = len(series)
            if n >= min_samples:
                if group_id not in valid_group_ids: # Avoid duplicates if order has repeats
                    valid_group_ids.append(group_id)
                    valid_group_data_map[group_id] = series
        else:
            for h_cat in hue_categories:
                group_id = (x_cat, h_cat)
                data_h_filtered = data_x_filtered[data_x_filtered[hue] == h_cat]
                if data_h_filtered.empty:
                    if h_cat not in data_x_filtered[hue].unique(): continue
                    else: series = pd.Series(dtype=float)
                else: series = data_h_filtered[y].dropna()
                n = len(series)
                if n >= min_samples:
                     if group_id not in valid_group_ids:
                         valid_group_ids.append(group_id)
                         valid_group_data_map[group_id] = series

    # Check if enough groups remain
    if len(valid_group_ids) < 3:
        # Clean up temporary column if created
        if group_col_name in data.columns: data.drop(columns=[group_col_name], inplace=True)
        raise ValueError(f"Requirement of at least 3 groups not met after filtering. Found {len(valid_group_ids)} valid groups: {valid_group_ids}")

    # Get list of series in the order of valid_group_ids for stats functions
    valid_group_data_list = [valid_group_data_map[gid] for gid in valid_group_ids]

    # --- Perform Brown-Forsythe Test ---
    bf_stat, bf_p_value = np.nan, np.nan
    try:
        bf_stat, bf_p_value = stats.levene(*valid_group_data_list, center='median')
    except Exception as e:
        print(f"Error during Brown-Forsythe test execution: {e}")
        if group_col_name in data.columns: data.drop(columns=[group_col_name], inplace=True)
        return None # Cannot proceed without homogeneity check

    # --- Choose and Perform Overall Comparison Test ---
    overall_test = "N/A"
    overall_stat = np.nan
    overall_p_value = np.nan
    posthoc_test = "N/A"
    posthoc_results = None

    # Filter original data for post-hoc tests using the combined group ID
    # Ensure the filter uses the correct group IDs (string or tuple)
    if hue:
        combined_group_ids_str = [f"{gid[0]}__{gid[1]}" for gid in valid_group_ids]
        posthoc_data = data[data[group_col_name].isin(combined_group_ids_str)].copy()
    else:
        posthoc_data = data[data[group_col_name].isin(valid_group_ids)].copy()

    # Drop NaNs from the relevant column *within the filtered data* for posthocs
    posthoc_data.dropna(subset=[y], inplace=True)

    # Check if enough data remains for posthocs after potential NaN drop
    if posthoc_data[group_col_name].nunique() < 2 or posthoc_data.empty:
         print("Warning: Insufficient data remaining for post-hoc tests after NaN removal.")
         bf_p_value = alpha + 1 # Ensure we don't try posthocs below

    # === Case 1: Variances are Homogeneous (Brown-Forsythe p > alpha) ===
    if bf_p_value > alpha:
        # Perform ANOVA
        warnings.warn("ANOVA assumes normality within groups, which was not explicitly checked before running based on user specification.", UserWarning)
        overall_test = "ANOVA"
        try:
            overall_stat, overall_p_value = stats.f_oneway(*valid_group_data_list)

            # Perform post-hoc if ANOVA is significant
            if overall_p_value < alpha:
                 posthoc_test = "Pairwise t-test (Bonferroni)"
                 try:
                     # Use scikit-posthocs t-test with Bonferroni correction
                     posthoc_results = sp.posthoc_ttest(posthoc_data, val_col=y, group_col=group_col_name, p_adjust='bonferroni', equal_var=True) # Assume equal var based on BF test
                 except Exception as e_ph:
                     print(f"Error during Bonferroni post-hoc t-test: {e_ph}")
                     posthoc_test = "Post-hoc Failed"
                     posthoc_results = None
            else:
                posthoc_test = "N/A (ANOVA p >= alpha)"

        except Exception as e:
            print(f"Error during ANOVA execution: {e}")
            overall_test = "ANOVA Failed"

    # === Case 2: Variances are Heterogeneous (Brown-Forsythe p <= alpha) ===
    else:
        # Perform Kruskal-Wallis
        overall_test = "Kruskal-Wallis"
        try:
            overall_stat, overall_p_value = stats.kruskal(*valid_group_data_list, nan_policy='omit') # Omit any remaining NaNs just in case

            # Perform post-hoc if Kruskal-Wallis is significant
            if overall_p_value < alpha:
                posthoc_test = "Dunn (Bonferroni)"
                try:
                     # Use scikit-posthocs Dunn test with Bonferroni correction
                     posthoc_results = sp.posthoc_dunn(posthoc_data, val_col=y, group_col=group_col_name, p_adjust='bonferroni')
                except Exception as e_ph:
                     print(f"Error during Dunn's post-hoc test: {e_ph}")
                     posthoc_test = "Post-hoc Failed"
                     posthoc_results = None
            else:
                posthoc_test = "N/A (Kruskal-Wallis p >= alpha)"

        except Exception as e:
            print(f"Error during Kruskal-Wallis execution: {e}")
            overall_test = "Kruskal-Wallis Failed"

    # Clean up temporary column
    if group_col_name in data.columns:
        data.drop(columns=[group_col_name], inplace=True)

    # --- Compile Results ---
    final_results = {
        'groups_tested': valid_group_ids,
        'n_groups_tested': len(valid_group_ids),
        'homogeneity_test': 'Brown-Forsythe',
        'homogeneity_stat': bf_stat,
        'homogeneity_p_value': bf_p_value,
        'overall_comparison_test': overall_test,
        'overall_stat': overall_stat,
        'overall_p_value': overall_p_value,
        'posthoc_test': posthoc_test,
        'posthoc_results': posthoc_results # This will be a DataFrame or None
    }

    return final_results