import pandas as pd
import pytest

polars = pytest.importorskip("polars")

from pycytominer import aggregate, annotate, consensus, feature_select, normalize


@pytest.fixture
def profile_data():
    profiles = pd.DataFrame({
        "Metadata_Plate": ["X", "X", "X", "Y"],
        "Metadata_Well": ["A01", "A01", "A02", "A02"],
        "Cells_a": [1, 3, 5, 7],
        "Nuclei_b": [2, 4, 6, 8],
    }).reset_index(drop=True)
    return profiles, polars.from_pandas(profiles)


def _assert_polars_matches(expected: pd.DataFrame, result):
    assert isinstance(result, polars.DataFrame)
    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_aggregate_returns_polars(profile_data):
    pd_profiles, pl_profiles = profile_data
    expected = aggregate(
        population_df=pd_profiles,
        strata=["Metadata_Plate", "Metadata_Well"],
        features="infer",
        operation="median",
    )

    result = aggregate(
        population_df=pl_profiles,
        strata=["Metadata_Plate", "Metadata_Well"],
        features="infer",
        operation="median",
    )

    _assert_polars_matches(expected, result)


def test_normalize_returns_polars(profile_data):
    pd_profiles, pl_profiles = profile_data
    features = ["Cells_a", "Nuclei_b"]
    meta_features = ["Metadata_Plate", "Metadata_Well"]

    expected = normalize(
        profiles=pd_profiles,
        features=features,
        meta_features=meta_features,
        samples="all",
        method="standardize",
    )

    result = normalize(
        profiles=pl_profiles,
        features=features,
        meta_features=meta_features,
        samples="all",
        method="standardize",
    )

    _assert_polars_matches(expected, result)


def test_feature_select_returns_polars(profile_data):
    pd_profiles, pl_profiles = profile_data
    features = ["Cells_a", "Nuclei_b"]

    expected = feature_select(
        profiles=pd_profiles,
        features=features,
        samples="all",
        operation="variance_threshold",
    )

    result = feature_select(
        profiles=pl_profiles,
        features=features,
        samples="all",
        operation="variance_threshold",
    )

    _assert_polars_matches(expected, result)


def test_annotate_returns_polars(profile_data):
    pd_profiles, pl_profiles = profile_data
    platemap = pd.DataFrame({
        "Metadata_Well": ["A01", "A02"],
        "Metadata_Treatment": ["ctl", "trt"],
    })

    expected = annotate(
        profiles=pd_profiles,
        platemap=platemap,
        join_on=["Metadata_Well", "Metadata_Well"],
        add_metadata_id_to_platemap=False,
    )

    result = annotate(
        profiles=pl_profiles,
        platemap=polars.from_pandas(platemap),
        join_on=["Metadata_Well", "Metadata_Well"],
        add_metadata_id_to_platemap=False,
    )

    _assert_polars_matches(expected, result)


def test_consensus_returns_polars(profile_data):
    pd_profiles, pl_profiles = profile_data

    expected = consensus(
        profiles=pd_profiles,
        replicate_columns=["Metadata_Plate", "Metadata_Well"],
        operation="median",
        features="infer",
    )

    result = consensus(
        profiles=pl_profiles,
        replicate_columns=["Metadata_Plate", "Metadata_Well"],
        operation="median",
        features="infer",
    )

    _assert_polars_matches(expected, result)
