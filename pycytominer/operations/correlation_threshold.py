"""
Returns list of features such that no two features have a correlation greater than a
specified threshold
"""

from typing import Union

import numpy as np
import pandas as pd

from pycytominer.cyto_utils.features import infer_cp_features
from pycytominer.cyto_utils.util import (
    _get_correlation_matrix,
    check_correlation_method,
)


def correlation_threshold(
    population_df: pd.DataFrame,
    features: Union[str, list[str]] = "infer",
    samples: str = "all",
    threshold: float = 0.9,
    method: str = "pearson",
) -> list[str]:
    """Exclude features that have correlations above a certain threshold

    Parameters
    ----------
    population_df : pd.DataFrame
        DataFrame that includes metadata and observation features.
    features : list, default "infer"
        A list of strings corresponding to feature measurement column names in the
        `population_df` DataFrame. All features listed must be found in `population_df`.
        Defaults to "infer". If "infer", then assume CellProfiler features are those
        prefixed with "Cells", "Nuclei", or "Cytoplasm".
    samples : str, default "all"
        List of samples to perform operation on. The function uses a pd.DataFrame.query()
        function, so you should  structure samples in this fashion. An example is
        "Metadata_treatment == 'control'" (include all quotes).
        If "all", use all samples to calculate.
    threshold - float, default 0.9
        Must be between (0, 1) to exclude features
    method - str, default "pearson"
        indicating which correlation metric to use to test cutoff

    Returns
    -------
    excluded_features : list of str
         List of features to exclude from the population_df.
    """

    # Checking if the provided correlation method is supported
    method = check_correlation_method(method)

    # Checking if the threshold is between 0 and 1
    if not 0 <= threshold <= 1:
        raise ValueError("threshold variable must be between (0 and 1)")

    # Subset dataframe and calculate correlation matrix across subset features
    # If samples is not 'all', then subset the dataframe
    if samples != "all":
        # Using pandas query to filter rows based on the conditions provided in the
        # samples parameter
        population_df = population_df.query(expr=samples)

    # Infer CellProfiler features if 'features' is set to 'infer'
    if features == "infer":
        # Infer CellProfiler features
        inferred_features = infer_cp_features(population_df)
    elif isinstance(features, list):
        inferred_features = features

    # Subset the DataFrame to only include the features of interest
    population_df = population_df.loc[:, inferred_features]

    data_cor_df = _get_correlation_matrix(population_df, method)

    # Get absolute sum of correlation across features
    # The lower the index, the less correlation to the full data frame
    # We want to drop features with highest correlation, so drop higher index
    variable_cor_sum = data_cor_df.abs().sum().sort_values().index

    # Find qualifying pairs directly in the lower triangle. This avoids materializing
    # a long DataFrame and applying a Python callback to every qualifying pair.
    pair_a_indices, pair_b_indices = np.nonzero(
        np.tril(data_cor_df.to_numpy() > threshold, k=-1)
    )

    if pair_a_indices.size == 0:
        return []

    # Convert the sorted feature names into an integer rank for vectorized lookup.
    correlation_ranks = variable_cor_sum.get_indexer(population_df.columns)
    excluded_indices = np.where(
        correlation_ranks[pair_a_indices] > correlation_ranks[pair_b_indices],
        pair_a_indices,
        pair_b_indices,
    )

    # Return each exclusion once, ordered consistently with the input features.
    return population_df.columns[np.unique(excluded_indices)].tolist()
